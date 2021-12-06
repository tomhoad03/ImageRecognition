import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class Classifier1 {
    private VFSListDataset<FImage> testing;
    private VFSGroupDataset<FImage> training;

    public Classifier1(VFSListDataset<FImage> testing, VFSGroupDataset<FImage> training) {
        this.testing = testing;
        this.training = training;
    }
}
