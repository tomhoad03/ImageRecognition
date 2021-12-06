import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

public class ImageRecognition {
    public static void main(String[] args) {
        MBFImage image = new MBFImage(400,70, ColourSpace.RGB);
        image.fill(RGBColour.WHITE);
        image.drawText("Hello Budapest!", 10, 60, HersheyFont.CURSIVE, 50, RGBColour.BLACK);
        DisplayUtilities.display(image);
    }
}