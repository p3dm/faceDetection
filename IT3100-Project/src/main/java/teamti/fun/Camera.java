package teamti.fun;


import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public class Camera {
    public static void main(String[] args) {
        boolean running = true;
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        final int RES_WIDTH = 1280;
        final int RES_HEIGHT = 960;

        VideoCapture capture = new VideoCapture(0);
        capture.set(CAP_PROP_FRAME_WIDTH, RES_WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, RES_HEIGHT);

        if (!capture.open(0)) {
            System.out.println("Can not open the cam !!!");
        }

        Mat coloring = new Mat();

        final JFrame mainframe = new JFrame("Face Detection");
        mainframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainframe.setSize(RES_WIDTH, RES_HEIGHT);
        mainframe.setLocationRelativeTo(null);
        mainframe.setVisible(true);

        JLabel label = new JLabel();

        mainframe.add(label);

        while (running) {
            while (capture.read(coloring) && mainframe.isVisible()) {
                FaceDetection.detectAndDraw(coloring);
                BufferedImage img = matToBufferedImage(coloring);
                ImageIcon icon = new ImageIcon(img);
                label.setIcon(icon);


            }
            if (!mainframe.isVisible()) {
                running = false;
            } else {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    running = false;
                }
            }
        }
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];
        mat.data().get(buffer); // get all the pixels
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }
}