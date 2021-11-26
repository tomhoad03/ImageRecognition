import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

public class Budapest {
    public static void main(String[] args) {
        // Create an image
        MBFImage image = new MBFImage(400,70, ColourSpace.RGB);

        // Fill the image with white
        image.fill(RGBColour.WHITE);

        // Render some test into the image - exercise 1
        image.drawText("Hello Budapest!", 10, 60, HersheyFont.CURSIVE, 50, RGBColour.BLACK);

        // Display the image
        DisplayUtilities.display(image);
    }
}