import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.*;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.FeatureVectorCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.ml.clustering.kmeans.FeatureVectorKMeans;
import org.openimaj.util.pair.IntDoublePair;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

public class Classifier2 {
    private final VFSGroupDataset<FImage> training;
    private final VFSListDataset<FImage> testing;

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
        System.out.println("Training the assigner from a sample of the training dataset...");
        HardAssigner<DoubleFV, float[], IntFloatPair> assigner2 = trainAssigner(GroupedUniformRandomisedSampler.sample(training, 20));

        System.out.println("Setting up the extractor and classifier...");
        Extractor2 extractor2 = new Extractor2(assigner2);
        LiblinearAnnotator liblinearAnnotator = new LiblinearAnnotator<>(extractor2, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L1R_L2LOSS_SVC, 15, 0.1, 0, true);

        System.out.println("Training the assigner from the training dataset...");
        liblinearAnnotator.train(training);

        System.out.println("Running the classifier on the testing dataset...");
    }

    // Extracts the features of an image to make a bag of visual words
    static class Extractor2 implements FeatureExtractor<DoubleFV, FImage> {
        private final HardAssigner<DoubleFV, float[], IntFloatPair> assigner2;

        public Extractor2(HardAssigner<DoubleFV, float[], IntFloatPair> assigner2) {
            this.assigner2 = assigner2;
        }

        @Override
        public DoubleFV extractFeature(FImage image) {
            BagOfVisualWords<DoubleFV> bovw = new BagOfVisualWords<>(assigner2);
            return bovw.aggregateVectorsRaw(extractPatchVectors(image)).normaliseFV();
        }
    }

    // Trains the assigner using a sample from the training dataset using K-Means clustering
    HardAssigner<DoubleFV, float[], IntFloatPair> trainAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> sample) {
        System.out.println("Extract patch vectors from sample set images...");

        List<DoubleFV> allVectors = new ArrayList<>();
        for (FImage image : sample) {
            allVectors.addAll(extractPatchVectors(image));
        }

        System.out.println("Perform K-means clustering on the patch vectors...");

        FeatureVectorKMeans<DoubleFV> kMeans = FeatureVectorKMeans.createExact(500, DoubleFVComparison.EUCLIDEAN);
        FeatureVectorCentroidsResult<DoubleFV> result = kMeans.cluster(allVectors);

        System.out.println("K-means clustering finished...");

        return result.defaultHardAssigner();
    }

    // Converts an image to a collection of vectors that represent 8x8 patches each spaced 4 pixels apart
    static ArrayList<DoubleFV> extractPatchVectors(FImage image) {
        ArrayList<DoubleFV> patchVectors = new ArrayList<>();

        for (int i = 0; i < image.getWidth(); i += 4) {
            for (int j = 0; j < image.getHeight(); j += 4) {
                FImage patch = image.extractROI(i, j, 8, 8);
                patchVectors.add(new DoubleFV(patch.normalise().getDoublePixelVector()));
            }
        }
        return patchVectors;
    }
}
