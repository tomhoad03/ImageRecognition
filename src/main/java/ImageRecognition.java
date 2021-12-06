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
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}