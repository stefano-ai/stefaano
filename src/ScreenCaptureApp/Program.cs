using System;
using System.Windows.Forms;

namespace ScreenCaptureApp
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            ApplicationConfiguration.Initialize();
            var config = AppConfig.Load();
            if (!FirstRunDialog.EnsureConsent(config))
            {
                return;
            }

            using var captureService = new CaptureService(config);
            using var tray = new TrayController(captureService);
            tray.Start();
            Application.Run();
        }
    }
}
