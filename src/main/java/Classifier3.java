import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class Classifier3 {
    private VFSListDataset<FImage> testing;
    private VFSGroupDataset<FImage> training;

    public Classifier3(VFSListDataset<FImage> testing, VFSGroupDataset<FImage> training) {
        this.testing = testing;
        this.training = training;
    }
}
