import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

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
    3. Extract bovw features from the image patches
    4. Use K-Means clustering to group the bags into 1 of 500 clusters
    5. Create a histogram from the clusters
    6. Classify the image from the histogram
     */

    public void run() {
        HardAssigner<byte[], float[], IntFloatPair> assigner2 = trainAssigner();
        Extractor2 extractor2 = new Extractor2(assigner2);

        LiblinearAnnotator liblinearAnnotator = new LiblinearAnnotator<>(extractor2, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L1R_L2LOSS_SVC, 15, 0.1);
        liblinearAnnotator.train(training);
    }

    private static class Extractor2 implements FeatureExtractor<DoubleFV, FImage> {
        private final HardAssigner<byte[], float[], IntFloatPair> assigner2;

        public Extractor2(HardAssigner<byte[], float[], IntFloatPair> assigner2) {
            this.assigner2 = assigner2;
        }

        @Override
        public DoubleFV extractFeature(FImage image) {
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner2);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 8, 8);

            return spatial.aggregate(null /*List of feature vectors*/, image.getBounds()).normaliseFV();
        }
    }

    HardAssigner<byte[], float[], IntFloatPair> trainAssigner() {
        return null;
    }
}
