import com.android.dx.rop.annotation.AnnotationsList;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.DisplayUtilities;
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
import java.util.stream.Stream;

public class Classifier1 extends BatchAnnotator<FImage, String> {
    private Set<String> annotations;
    private FloatNearestNeighboursExact knn;
    private String[] trainingAnnotations;
    private int k;

    public Classifier1(int k){
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
            System.out.println(instance);
            for (FImage image : training.get(instance)) {
                trainingData[imageIndex] = flattenImage(makeTiny(image));
                trainingAnnotations[imageIndex] = instance;
                imageIndex++;
            }

        }

        this.knn = new FloatNearestNeighboursExact(trainingData);


    }

    @Override
    public void train(List<? extends Annotated<FImage, String>> list) {

    }
}
