import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.nio.file.Paths;
import java.util.Map;

public class ImageRecognition {
    public static void main(String[] args) {
        try {
            final VFSGroupDataset<FImage> training = new VFSGroupDataset<>(Paths.get("").toAbsolutePath() + "\\images\\training\\training", ImageUtilities.FIMAGE_READER);
            final VFSGroupDataset<FImage> testing = new VFSGroupDataset<>(Paths.get("").toAbsolutePath() + "\\images\\testing", ImageUtilities.FIMAGE_READER);

            Classifier1 classifier1 = new Classifier1(10);
            classifier1.train(training);

            ClassificationEvaluator<CMResult<String>, String, FImage> evaluator = new ClassificationEvaluator<>(classifier1, testing, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
            Map<FImage, ClassificationResult<String>> evaluation = evaluator.evaluate();

            //Classifier2 classifier2 = new Classifier2(testing, training);
            //Classifier3 classifier3 = new Classifier3(testing, training);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}