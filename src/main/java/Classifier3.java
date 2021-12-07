import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;

public class Classifier3 {
    private VFSGroupDataset<FImage> testing;
    private VFSGroupDataset<FImage> training;

    public Classifier3(VFSGroupDataset<FImage> testing, VFSGroupDataset<FImage> training) {
        this.testing = testing;
        this.training = training;
    }
}
