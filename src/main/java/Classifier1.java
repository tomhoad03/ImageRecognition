import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;

public class Classifier1 {
    private VFSGroupDataset<FImage> testing;
    private VFSGroupDataset<FImage> training;

    public Classifier1(VFSGroupDataset<FImage> testing, VFSGroupDataset<FImage> training) {
        this.testing = testing;
        this.training = training;
    }
}
