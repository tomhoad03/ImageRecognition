import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FeatureVectorCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FeatureVectorKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.*;

public class Classifier2 {
    private final VFSGroupDataset<FImage> training;
    private final VFSGroupDataset<FImage> testing;

    public Classifier2(VFSGroupDataset<FImage> training, VFSGroupDataset<FImage> testing) {
        this.training = training;
        this.testing = testing;
    }

    public void run() throws Exception {
        System.out.println("Training the assigner from a sample of the training dataset...");
        HardAssigner<FloatFV, float[], IntFloatPair> assigner2 = trainAssigner(GroupedUniformRandomisedSampler.sample(training, 50));

        System.out.println("Setting up the extractor and classifier...");
        Extractor2 extractor2 = new Extractor2(assigner2);
        LiblinearAnnotator<FImage, String> liblinearAnnotator = new LiblinearAnnotator<>(extractor2, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L1R_L2LOSS_SVC, 1, 0.00001);

        System.out.println("Running the assigner on the training dataset...");
        liblinearAnnotator.train(training);

        System.out.println("Running the classifier on the testing dataset...");
        ClassificationEvaluator<CMResult<String>, String, FImage> evaluator2 = new ClassificationEvaluator<>(liblinearAnnotator, testing, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> evaluation2 = evaluator2.evaluate();

        System.out.println("Analysing the results of the classification...");
        File files = new File(Paths.get("").toAbsolutePath() + "\\images\\testing\\testing");
        ArrayList<String> fileNames = new ArrayList<>(List.of(Objects.requireNonNull(files.list())));
        ArrayList<String> results = new ArrayList<>(); int count = 0;

        for (FImage image : testing.get("testing")) {
            for (Map.Entry<FImage, ClassificationResult<String>> evalEntry : evaluation2.entrySet()) {
                if (evalEntry.getKey().equals(image)) {
                    String fileName = fileNames.get(count); count++;
                    String prediction = evalEntry.getValue().getPredictedClasses().toString();

                    results.add(fileName + " " + prediction + "\n");
                    break;
                }
            }
        }
        results.sort(Comparator.comparing(o -> Integer.parseInt(o.substring(0, o.indexOf(".jpg")))));

        System.out.println("Printing the results of the classification...");
        FileWriter fileWriter = new FileWriter(Paths.get("").toAbsolutePath() + "\\runs\\run2.txt");
        for (String result : results) {
            fileWriter.write(result);
        }
        fileWriter.close();
        System.out.println("Classification has finished...");
    }

    // Extracts the features of an image to make a bag of visual words
    static class Extractor2 implements FeatureExtractor<SparseIntFV, FImage> {
        private final HardAssigner<FloatFV, float[], IntFloatPair> assigner2;

        public Extractor2(HardAssigner<FloatFV, float[], IntFloatPair> assigner2) {
            this.assigner2 = assigner2;
        }

        @Override
        public SparseIntFV extractFeature(FImage image) {
            BagOfVisualWords<FloatFV> bovw = new BagOfVisualWords<>(assigner2);
            return bovw.aggregateVectorsRaw(extractPatchVectors(image));
        }
    }

    // Trains the assigner using a sample from the training dataset using K-Means clustering
    HardAssigner<FloatFV, float[], IntFloatPair> trainAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> sample) {
        System.out.println("Extract patch vectors from sample set images...");
        List<FloatFV> allVectors = new ArrayList<>();
        for (FImage image : sample) {
            allVectors.addAll(extractPatchVectors(image));
        }

        System.out.println("Perform K-means clustering on the patch vectors...");
        FeatureVectorKMeans<FloatFV> kMeans = FeatureVectorKMeans.createExact(500, FloatFVComparison.EUCLIDEAN);
        FeatureVectorCentroidsResult<FloatFV> result = kMeans.cluster(allVectors);
        return result.defaultHardAssigner();
    }

    // Converts an image to a collection of vectors that represent 8x8 patches each spaced 4 pixels apart
    static ArrayList<FloatFV> extractPatchVectors(FImage image) {
        ArrayList<FloatFV> patchVectors = new ArrayList<>();

        for (int i = 0; i < image.getWidth() - 4; i += 4) {
            for (int j = 0; j < image.getHeight() - 4; j += 4) {
                FImage patch = image.extractROI(i, j, 8, 8);
                patchVectors.add(new FloatFV(patch.normalise().getFloatPixelVector()));
            }
        }
        return patchVectors;
    }
}
