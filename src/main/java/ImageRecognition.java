import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.nio.file.Paths;

public class ImageRecognition {
    public static void main(String[] args) {
        try {
            VFSListDataset<FImage> testing = new VFSListDataset<>("zip:" + Paths.get("").toAbsolutePath() + "\\images\\testing.zip", ImageUtilities.FIMAGE_READER);
            VFSListDataset<FImage> training = new VFSListDataset<>("zip:" + Paths.get("").toAbsolutePath() + "\\images\\training.zip", ImageUtilities.FIMAGE_READER);

            DisplayUtilities.display("Testing", testing);
            DisplayUtilities.display("Training", training);

            Classifier1 classifier1 = new Classifier1(testing, training);
            Classifier2 classifier2 = new Classifier2(testing, training);
            Classifier3 classifier3 = new Classifier3(testing, training);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}