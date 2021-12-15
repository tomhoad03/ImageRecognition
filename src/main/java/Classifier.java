import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;

import java.util.Map;

public class Classifier {
    private final VFSGroupDataset<FImage> training;
    private final VFSGroupDataset<FImage> testing;
    private Map<FImage, ClassificationResult<String>> evaluation;

    public Classifier(VFSGroupDataset<FImage> training, VFSGroupDataset<FImage> testing) {
        this.training = training;
        this.testing = testing;
    }

    public VFSGroupDataset<FImage> getTraining() {
        return training;
    }

    public VFSGroupDataset<FImage> getTesting() {
        return testing;
    }

    public Map<FImage, ClassificationResult<String>> getEvaluation() {
        return evaluation;
    }

    public void setEvaluation(Map<FImage, ClassificationResult<String>> evaluation) {
        this.evaluation = evaluation;
    }
}
