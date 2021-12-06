import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.BatchAnnotator;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.util.array.ArrayUtils;

import java.util.List;
import java.util.Set;

public class Classifier1 extends BatchAnnotator<FImage, String> {
    private VFSGroupDataset<FImage> testing;
    private VFSGroupDataset<FImage> training;
    private Set<String> annotations;

    public Classifier1(VFSGroupDataset<FImage> testing, VFSGroupDataset<FImage> training) {
        this.testing = testing;
        this.training = training;
        this.annotations = training.getGroups();
    }

    public void run() {

        float[][] trainingData = new float[training.numInstances()][16*16];

        int imageIndex = 0;
        for (String instance : training.getGroups()) {
            for (FImage image : training.get(instance)) {
                trainingData[imageIndex] = flattenImage(makeTiny(image));
                imageIndex++;
            }

        }

        FloatNearestNeighboursExact knn = new FloatNearestNeighboursExact(trainingData);


        for (String instance : testing.getGroups()) {
            for (FImage image : testing.get(instance)) {
                knn.searchKNN(flattenImage(makeTiny(image)), 10);
            }
        }





        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        knn, testing, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);


    }

    public FImage makeTiny(FImage image) {
        int size = Math.min(image.width, image.height);

        image = ResizeProcessor.resample(image.extractCenter(size, size), 16, 16);

        return image.normalise();
    }

    public float[] flattenImage(FImage image) {
        return ArrayUtils.concatenate(image.pixels);
    }

    @Override
    public Set<String> getAnnotations() {
        return this.annotations;
    }

    @Override
    public List<ScoredAnnotation<String>> annotate(FImage object) {
        return null;
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training) {

        this.annotations = training.getGroups();
        float[][] trainingData = new float[training.numInstances()][16*16];

        int imageIndex = 0;
        for (String instance : training.getGroups()) {
            for (FImage image : training.get(instance)) {
                trainingData[imageIndex] = flattenImage(makeTiny(image));
                imageIndex++;
            }

        }

        FloatNearestNeighboursExact knn = new FloatNearestNeighboursExact(trainingData);


    }

    @Override
    public void train(List<? extends Annotated<FImage, String>> list) {

    }
}
