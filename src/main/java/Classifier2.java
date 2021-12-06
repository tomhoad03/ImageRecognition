import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

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
        HardAssigner<byte[], float[], IntFloatPair> assigner2 = trainAssigner(GroupedUniformRandomisedSampler.sample(training, 20));
        Extractor2 extractor2 = new Extractor2(assigner2);

        LiblinearAnnotator liblinearAnnotator = new LiblinearAnnotator<>(extractor2, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L1R_L2LOSS_SVC, 15, 0.1);
        liblinearAnnotator.train(training);
    }

    static class Extractor2 implements FeatureExtractor<DoubleFV, FImage> {
        private final HardAssigner<byte[], float[], IntFloatPair> assigner2;

        public Extractor2(HardAssigner<byte[], float[], IntFloatPair> assigner2) {
            this.assigner2 = assigner2;
        }

        @Override
        public DoubleFV extractFeature(FImage image) {
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner2);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 1, 1);

            return spatial.aggregate(imageFeatures(image), image.getBounds()).normaliseFV();
        }
    }

    HardAssigner<byte[], float[], IntFloatPair> trainAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> sample) {
        List<LocalFeatureList<Keypoint>> allkeys = new ArrayList<>();

        for (FImage image : sample) {
            allkeys.add(imageFeatures(image));
        }

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    static LocalFeatureList<Keypoint> imageFeatures(FImage image) {
        return null;
    }
}
