import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class Classifier1 {
    private VFSListDataset<FImage> testing;
    private VFSListDataset<FImage> training;

    public Classifier1(VFSListDataset<FImage> testing, VFSListDataset<FImage> training) {
        this.testing = testing;
        this.training = training;
    }
}
