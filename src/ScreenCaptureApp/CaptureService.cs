using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Linq;

namespace ScreenCaptureApp
{
    public class CaptureService
    {
        private readonly AppConfig _config;
        private readonly List<string> _normalizedBlockList;

        public CaptureService(AppConfig config)
        {
            _config = config;
            _normalizedBlockList = config.BlockedProcesses.Select(p => p.ToLowerInvariant()).ToList();
        }

        public bool ShouldCaptureCurrentWindow()
        {
            var foreground = NativeMethods.GetForegroundWindow();
            if (foreground == IntPtr.Zero)
            {
                return true;
            }

            _ = NativeMethods.GetWindowThreadProcessId(foreground, out var pid);
            try
            {
                var process = System.Diagnostics.Process.GetProcessById((int)pid);
                var name = process.ProcessName.ToLowerInvariant();
                return !_normalizedBlockList.Contains(name);
            }
            catch
            {
                return true;
            }
        }

        public Bitmap CaptureScreen()
        {
            var primary = Screen.PrimaryScreen ?? throw new InvalidOperationException("No screens detected");
            var bounds = primary.Bounds;
            var bitmap = new Bitmap(bounds.Width, bounds.Height, PixelFormat.Format32bppArgb);
            using var graphics = Graphics.FromImage(bitmap);
            graphics.CopyFromScreen(bounds.Location, Point.Empty, bounds.Size);
            ApplyBlurMasks(bitmap, _config.BlurMasks);
            return bitmap;
        }

        private static void ApplyBlurMasks(Bitmap bitmap, IEnumerable<BlurMask> masks)
        {
            foreach (var mask in masks)
            {
                var rectangle = mask.ToRectangle();
                using var region = bitmap.Clone(rectangle, bitmap.PixelFormat);
                var blurred = ApplyGaussianBlur(region, mask.Sigma);
                using var graphics = Graphics.FromImage(bitmap);
                graphics.DrawImage(blurred, rectangle.Location);
            }
        }

        private static Bitmap ApplyGaussianBlur(Bitmap source, double sigma)
        {
            var radius = (int)Math.Ceiling(sigma * 3);
            var size = radius * 2 + 1;
            var kernel = BuildKernel(size, sigma);
            var blurred = new Bitmap(source.Width, source.Height);

            for (int y = 0; y < source.Height; y++)
            {
                for (int x = 0; x < source.Width; x++)
                {
                    double r = 0, g = 0, b = 0, a = 0, weightSum = 0;
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int sampleX = Math.Clamp(x + kx, 0, source.Width - 1);
                            int sampleY = Math.Clamp(y + ky, 0, source.Height - 1);
                            var weight = kernel[ky + radius, kx + radius];
                            var pixel = source.GetPixel(sampleX, sampleY);
                            r += pixel.R * weight;
                            g += pixel.G * weight;
                            b += pixel.B * weight;
                            a += pixel.A * weight;
                            weightSum += weight;
                        }
                    }

                    if (weightSum > 0)
                    {
                        r /= weightSum;
                        g /= weightSum;
                        b /= weightSum;
                        a /= weightSum;
                    }

                    var color = Color.FromArgb((int)a.Clamp(0, 255), (int)r.Clamp(0, 255), (int)g.Clamp(0, 255), (int)b.Clamp(0, 255));
                    blurred.SetPixel(x, y, color);
                }
            }

            return blurred;
        }

        private static double[,] BuildKernel(int size, double sigma)
        {
            var kernel = new double[size, size];
            int radius = size / 2;
            double twoSigmaSquared = 2 * sigma * sigma;
            double normalization = 1.0 / (Math.PI * twoSigmaSquared);

            for (int y = -radius; y <= radius; y++)
            {
                for (int x = -radius; x <= radius; x++)
                {
                    double exponent = -((x * x + y * y) / twoSigmaSquared);
                    kernel[y + radius, x + radius] = normalization * Math.Exp(exponent);
                }
            }

            return kernel;
        }
    }

    internal static class NativeMethods
    {
        [DllImport("user32.dll")]
        public static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll")]
        public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
    }

    internal static class NumericExtensions
    {
        public static double Clamp(this double value, double min, double max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }
    }
}
