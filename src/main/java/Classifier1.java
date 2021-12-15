import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.BatchAnnotator;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Classifier1 extends Classifier {
    private final Annotator1 annotator1;

    public Classifier1(VFSGroupDataset<FImage> training, VFSGroupDataset<FImage> testing) {
        super(training, testing);
        this.annotator1 = new Annotator1(10);
    }

    public void run() {
        annotator1.train(getTraining());
        ClassificationEvaluator<CMResult<String>, String, FImage> evaluator = new ClassificationEvaluator<>(annotator1, getTesting(), new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        setEvaluation(evaluator.evaluate());
    }
}

class Annotator1 extends BatchAnnotator<FImage, String> {
    private Set<String> annotations;
    private FloatNearestNeighboursExact knn;
    private String[] trainingAnnotations;
    private final int k;

    public Annotator1(int k){
        this.k = k;
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
    public List<ScoredAnnotation<String>> annotate(FImage image) {
        List<IntFloatPair> nearest = this.knn.searchKNN(flattenImage(makeTiny(image)), this.k);
        List<ScoredAnnotation<String>> result = new ArrayList<ScoredAnnotation<String>>(1);

        nearest.stream()
                .map(e -> trainingAnnotations[e.getFirst()])
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
                .entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .ifPresent(e -> result.add(new ScoredAnnotation<String>(e.getKey(), (float) e.getValue() / this.k)));

        return result;
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training) {
        this.annotations = training.getGroups();
        float[][] trainingData = new float[training.numInstances()][16*16];
        this.trainingAnnotations = new String[trainingData.length];

        int imageIndex = 0;
        for (String instance : training.getGroups()) {
            for (FImage image : training.get(instance)) {
                trainingData[imageIndex] = flattenImage(makeTiny(image));
                trainingAnnotations[imageIndex] = instance;
                imageIndex++;
            }
        }
        this.knn = new FloatNearestNeighboursExact(trainingData);
    }

    @Override
    public void train(List<? extends Annotated<FImage, String>> list) { }
}