using System.Drawing;
using System.Windows.Forms;

namespace ScreenCaptureApp
{
    public class TrayController : IDisposable
    {
        private readonly NotifyIcon _trayIcon;
        private readonly CaptureService _captureService;
        private readonly Timer _captureTimer;
        private bool _paused;

        public TrayController(CaptureService captureService)
        {
            _captureService = captureService;
            _trayIcon = BuildTrayIcon();
            _captureTimer = new Timer { Interval = 2000 };
            _captureTimer.Tick += (_, _) => OnCaptureTick();
        }

        public void Start()
        {
            _paused = false;
            UpdateIndicator();
            _captureTimer.Start();
        }

        private void OnCaptureTick()
        {
            if (_paused)
            {
                return;
            }

            if (!_captureService.ShouldCaptureCurrentWindow())
            {
                _trayIcon.Text = "Capturing (skipped)";
                return;
            }

            using var bitmap = _captureService.CaptureScreen();
            // Encoding to disk or network would happen here.
            _trayIcon.Text = "Capturing";
        }

        private NotifyIcon BuildTrayIcon()
        {
            var contextMenu = new ContextMenuStrip();
            var pauseResume = new ToolStripMenuItem("Pause") { CheckOnClick = false };
            pauseResume.Click += (_, _) => TogglePause(pauseResume);
            contextMenu.Items.Add(pauseResume);
            contextMenu.Items.Add(new ToolStripMenuItem("Exit", null, (_, _) => Application.Exit()));

            return new NotifyIcon
            {
                Icon = SystemIcons.Information,
                Text = "Capturing",
                Visible = true,
                ContextMenuStrip = contextMenu
            };
        }

        private void TogglePause(ToolStripMenuItem toggleItem)
        {
            _paused = !_paused;
            toggleItem.Text = _paused ? "Resume" : "Pause";
            UpdateIndicator();
        }

        private void UpdateIndicator()
        {
            _trayIcon.Text = _paused ? "Capture paused" : "Capturing";
        }

        public void Dispose()
        {
            _captureTimer.Dispose();
            _trayIcon.Dispose();
        }
    }
}
