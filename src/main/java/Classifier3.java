import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;

public class Classifier3 {
    private VFSGroupDataset<FImage> training;
    private VFSGroupDataset<FImage> testing;

    public Classifier3(VFSGroupDataset<FImage> training, VFSGroupDataset<FImage> testing) {
        this.training = training;
        this.testing = testing;
    }

    public void run() throws Exception {

    }
}
