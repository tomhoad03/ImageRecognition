import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.nio.file.Paths;

public class ImageRecognition {
    public static void main(String[] args) {
        try {
            VFSGroupDataset<FImage> training = new VFSGroupDataset<>(Paths.get("").toAbsolutePath() + "\\images\\training", ImageUtilities.FIMAGE_READER);
            VFSGroupDataset<FImage> testing = new VFSGroupDataset<>(Paths.get("").toAbsolutePath() + "\\images\\testing", ImageUtilities.FIMAGE_READER);

            Classifier1 classifier1 = new Classifier1(training, testing);
            classifier1.run();

            Classifier2 classifier2 = new Classifier2(training, testing);
            classifier2.run();

            Classifier3 classifier3 = new Classifier3(training, testing);
            classifier3.run();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}