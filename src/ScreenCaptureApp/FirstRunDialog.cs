using System.Diagnostics;
using System.Windows.Forms;

namespace ScreenCaptureApp
{
    public static class FirstRunDialog
    {
        public static bool EnsureConsent(AppConfig config)
        {
            if (config.ConsentAccepted)
            {
                return true;
            }

            var message = "Screen capture requires your consent. Please review the privacy terms before continuing.";
            var result = MessageBox.Show(message, "Privacy and Consent", MessageBoxButtons.OKCancel, MessageBoxIcon.Information, MessageBoxDefaultButton.Button1, MessageBoxOptions.DefaultDesktopOnly, false);

            if (result == DialogResult.OK)
            {
                ShowPrivacyPolicy(config.PrivacyPolicyUrl);
                var confirm = MessageBox.Show("Do you accept the privacy terms and consent to capture?", "Confirm consent", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                if (confirm == DialogResult.Yes)
                {
                    config.ConsentAccepted = true;
                    config.Save();
                    return true;
                }
            }

            return false;
        }

        private static void ShowPrivacyPolicy(string url)
        {
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = url,
                    UseShellExecute = true
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Unable to open privacy policy: {ex.Message}");
            }
        }
    }
}
