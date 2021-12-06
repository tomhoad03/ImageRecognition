import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.*;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.model.patch.HistogramPatchModel;
import org.openimaj.image.model.patch.PatchClassificationModel;
import org.openimaj.image.processor.GridProcessor;
import org.openimaj.math.statistics.distribution.Histogram;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.ml.clustering.kmeans.FeatureVectorKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntDoublePair;
import org.openimaj.util.pair.IntFloatPair;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Classifier2 {
    private VFSGroupDataset<FImage> training;
    private VFSListDataset<FImage> testing;

    public Classifier2(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {
        this.training = training;
        this.testing = testing;
    }

    /*
    1. Take fixed size densely-sampled pixel patches (8x8 in size and 4 pixels apart in the x and y directions)
    2. Mean centering and normalise the patches
    3. Extract visual words features from the image patches
    4. Use K-Means clustering to group the bags into 1 of 500 clusters
    5. Create a BOVW from the clusters
    6. Classify the image from the BOVW
     */

    public void run() {
        HardAssigner<double[], double[], IntDoublePair> assigner2 = trainAssigner(GroupedUniformRandomisedSampler.sample(training, 20));
        Extractor2 extractor2 = new Extractor2(assigner2);

        LiblinearAnnotator liblinearAnnotator = new LiblinearAnnotator<>(extractor2, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L1R_L2LOSS_SVC, 15, 0.1, 0, true);
        liblinearAnnotator.train(training);
    }

    static class Extractor2 implements FeatureExtractor<DoubleFV, FImage> {
        private final HardAssigner<double[], double[], IntDoublePair> assigner2;

        public Extractor2(HardAssigner<double[], double[], IntDoublePair> assigner2) {
            this.assigner2 = assigner2;
        }

        @Override
        public DoubleFV extractFeature(FImage image) {
            BagOfVisualWords<double[]> bovw = new BagOfVisualWords<>(assigner2);

            return bovw.aggregateVectorsRaw(extractPatchVectors(image)).normaliseFV();
        }
    }

    HardAssigner<double[], double[], IntDoublePair> trainAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> sample) {
        ArrayList<double[]> allVectors = new ArrayList<>();

        for (FImage image : sample) {
            allVectors.addAll(extractPatchVectors(image));
        }
        double[][] allDoubleVectors = new double[0][];

        for (int i = 0; i < allVectors.size(); i++) {
            allDoubleVectors[i] = allVectors.get(i);
        }

        DoubleKMeans kMeans = DoubleKMeans.createExact(500);
        DoubleCentroidsResult result = kMeans.cluster(allDoubleVectors);

        return result.defaultHardAssigner();
    }

    static ArrayList<double[]> extractPatchVectors(FImage image) {
        ArrayList<double[]> patchVectors = new ArrayList<>();

        for (int i = 0; i < image.getWidth(); i += 4) {
            for (int j = 0; j < image.getHeight(); i += 4) {
                FImage patch = image.extractROI(i, j, 8, 8);
                patchVectors.add(patch.normalise().getDoublePixelVector());
            }
        }
        return patchVectors;
    }
}
