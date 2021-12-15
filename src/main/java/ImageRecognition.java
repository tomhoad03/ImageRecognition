import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.*;

public class ImageRecognition {
    public static void main(String[] args) {
        try {
            final VFSGroupDataset<FImage> training = new VFSGroupDataset<>(Paths.get("").toAbsolutePath() + "\\images\\training", ImageUtilities.FIMAGE_READER);
            final VFSGroupDataset<FImage> testing = new VFSGroupDataset<>(Paths.get("").toAbsolutePath() + "\\images\\testing", ImageUtilities.FIMAGE_READER);

            Classifier1 classifier1 = new Classifier1(training, testing);
            classifier1.run();
            print(classifier1);

            Classifier2 classifier2 = new Classifier2(training, testing);
            classifier2.run();
            print(classifier2);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void print(Classifier classifier) throws Exception {
        System.out.println("Analysing the results of the classification...");
        File files = new File(Paths.get("").toAbsolutePath() + "\\images\\testing\\testing");
        ArrayList<String> fileNames = new ArrayList<>(List.of(Objects.requireNonNull(files.list())));
        ArrayList<String> results = new ArrayList<>(); int count = 0;

        for (FImage image : classifier.getTesting().get("testing")) {
            for (Map.Entry<FImage, ClassificationResult<String>> evalEntry : classifier.getEvaluation().entrySet()) {
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
}