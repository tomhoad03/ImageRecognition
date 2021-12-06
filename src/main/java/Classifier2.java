import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.lang.annotation.Annotation;

public class Classifier2 {
    private VFSGroupDataset<FImage> training;
    private VFSListDataset<FImage> testing;

    public Classifier2(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {
        this.training = training;
        this.testing = testing;
    }

    public void run() {
        HardAssigner<byte[], float[], IntFloatPair> assigner2 = trainAssigner();
        Extractor2 extractor2 = new Extractor2(assigner2);

        LiblinearAnnotator liblinearAnnotator = new LiblinearAnnotator<>(extractor2, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L1R_L2LOSS_SVC, 15, 0.1);
        liblinearAnnotator.train(training);

        for (FImage image : testing) {
            System.out.println(liblinearAnnotator.annotate(image));
        }
    }

    private static class Extractor2 implements FeatureExtractor<FeatureVector, FImage> {
        private final HardAssigner<byte[], float[], IntFloatPair> assigner2;

        public Extractor2(HardAssigner<byte[], float[], IntFloatPair> assigner2) {
            this.assigner2 = assigner2;
        }

        @Override
        public FeatureVector extractFeature(FImage image) {
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner2);

            return null;
        }
    }

    HardAssigner<byte[], float[], IntFloatPair> trainAssigner() {
        return null;
    }
}
