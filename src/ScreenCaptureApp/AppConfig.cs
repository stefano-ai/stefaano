using System.Drawing;
using System.Text.Json;

namespace ScreenCaptureApp
{
    public class AppConfig
    {
        private const string ConfigFileName = "capture_config.json";

        public bool ConsentAccepted { get; set; }

        public string PrivacyPolicyUrl { get; set; } = "https://example.com/privacy";

        public List<string> BlockedProcesses { get; set; } = new();

        public List<BlurMask> BlurMasks { get; set; } = new();

        public static string GetConfigPath()
        {
            var folder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ScreenCaptureApp");
            Directory.CreateDirectory(folder);
            return Path.Combine(folder, ConfigFileName);
        }

        public static AppConfig Load()
        {
            var path = GetConfigPath();
            if (!File.Exists(path))
            {
                return new AppConfig();
            }

            var content = File.ReadAllText(path);
            var config = JsonSerializer.Deserialize<AppConfig>(content);
            return config ?? new AppConfig();
        }

        public void Save()
        {
            var path = GetConfigPath();
            var content = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, content);
        }
    }

    public class BlurMask
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public double Sigma { get; set; } = 6.0;

        public Rectangle ToRectangle() => new(X, Y, Width, Height);
    }
}
