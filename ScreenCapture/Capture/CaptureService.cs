using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Graphics.Capture;
using Windows.Graphics.DirectX.Direct3D11;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;

namespace ScreenCapture.Capture;

public sealed class CaptureService : IAsyncDisposable
{
    private readonly TimeSpan _defaultInterval = TimeSpan.FromSeconds(1);
    private readonly object _syncRoot = new();
    private IDirect3DDevice? _direct3DDevice;
    private Direct3D11CaptureFramePool? _framePool;
    private GraphicsCaptureSession? _session;
    private GraphicsCaptureItem? _captureItem;
    private SizeInt32 _lastSize;

    public bool IsGraphicsCaptureAvailable => GraphicsCaptureSession.IsSupported();

    public async Task<IRandomAccessStream?> CaptureOnceAsync(CancellationToken cancellationToken = default)
    {
        if (IsGraphicsCaptureAvailable)
        {
            await EnsureCaptureSessionAsync(cancellationToken).ConfigureAwait(false);
            return await CaptureCurrentFrameAsync(cancellationToken).ConfigureAwait(false);
        }

        return await CaptureWithBitBltAsync().ConfigureAwait(false);
    }

    public async Task StartContinuousCaptureAsync(
        Func<IRandomAccessStream, Task> frameHandler,
        TimeSpan? interval = null,
        CancellationToken cancellationToken = default)
    {
        if (frameHandler is null)
        {
            throw new ArgumentNullException(nameof(frameHandler));
        }

        var delay = interval ?? _defaultInterval;

        if (IsGraphicsCaptureAvailable)
        {
            await EnsureCaptureSessionAsync(cancellationToken).ConfigureAwait(false);

            while (!cancellationToken.IsCancellationRequested)
            {
                var stream = await CaptureCurrentFrameAsync(cancellationToken).ConfigureAwait(false);
                if (stream != null)
                {
                    await frameHandler(stream).ConfigureAwait(false);
                }

                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }
        else
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var stream = await CaptureWithBitBltAsync().ConfigureAwait(false);
                if (stream != null)
                {
                    await frameHandler(stream).ConfigureAwait(false);
                }

                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        await Task.Yield();
        lock (_syncRoot)
        {
            _session?.Dispose();
            _framePool?.Dispose();
            _session = null;
            _framePool = null;
            _captureItem = null;
        }
    }

    private async Task EnsureCaptureSessionAsync(CancellationToken cancellationToken)
    {
        if (_session != null && _captureItem != null)
        {
            return;
        }

        var picker = new GraphicsCapturePicker();
        _captureItem = await picker.PickSingleItemAsync().AsTask(cancellationToken).ConfigureAwait(false);
        if (_captureItem == null)
        {
            throw new InvalidOperationException("No capture target selected.");
        }

        _direct3DDevice ??= Direct3D11Helper.CreateDevice();
        _lastSize = _captureItem.Size;
        _framePool = Direct3D11CaptureFramePool.Create(
            _direct3DDevice,
            Windows.Graphics.DirectX.DirectXPixelFormat.B8G8R8A8UIntNormalized,
            1,
            _lastSize);

        _session = _framePool.CreateCaptureSession(_captureItem);
        _session.StartCapture();
    }

    private async Task<IRandomAccessStream?> CaptureCurrentFrameAsync(CancellationToken cancellationToken)
    {
        if (_framePool == null)
        {
            return null;
        }

        var tcs = new TaskCompletionSource<IRandomAccessStream?>();
        void OnFrameArrived(Direct3D11CaptureFramePool sender, object args)
        {
            try
            {
                using var frame = sender.TryGetNextFrame();
                if (frame == null)
                {
                    tcs.TrySetResult(null);
                    return;
                }

                if (frame.ContentSize.Width != _lastSize.Width || frame.ContentSize.Height != _lastSize.Height)
                {
                    ResizeFramePool(frame.ContentSize);
                }

                var bitmap = SoftwareBitmap.CreateCopyFromSurfaceAsync(frame.Surface).AsTask().Result;
                var stream = new InMemoryRandomAccessStream();
                var encoder = BitmapEncoder.CreateAsync(BitmapEncoder.JpegEncoderId, stream).AsTask().Result;
                encoder.SetSoftwareBitmap(bitmap);
                encoder.IsThumbnailGenerated = false;
                encoder.FlushAsync().AsTask().Wait();
                stream.Seek(0);
                tcs.TrySetResult(stream);
            }
            catch (Exception ex)
            {
                tcs.TrySetException(ex);
            }
            finally
            {
                sender.FrameArrived -= OnFrameArrived;
            }
        }

        _framePool.FrameArrived += OnFrameArrived;
        using var registration = cancellationToken.Register(() => tcs.TrySetCanceled(cancellationToken));
        return await tcs.Task.ConfigureAwait(false);
    }

    private void ResizeFramePool(SizeInt32 newSize)
    {
        if (_framePool == null || _captureItem == null || _direct3DDevice == null)
        {
            return;
        }

        _framePool.Recreate(
            _direct3DDevice,
            Windows.Graphics.DirectX.DirectXPixelFormat.B8G8R8A8UIntNormalized,
            1,
            newSize);
        _lastSize = newSize;
    }

    private Task<IRandomAccessStream?> CaptureWithBitBltAsync()
    {
        var screenDc = Gdi32.GetDC(IntPtr.Zero);
        try
        {
            var width = Gdi32.GetSystemMetrics(Gdi32.SystemMetric.SM_CXSCREEN);
            var height = Gdi32.GetSystemMetrics(Gdi32.SystemMetric.SM_CYSCREEN);
            var compatibleDc = Gdi32.CreateCompatibleDC(screenDc);
            var bitmap = Gdi32.CreateCompatibleBitmap(screenDc, width, height);
            var old = Gdi32.SelectObject(compatibleDc, bitmap);

            _ = Gdi32.BitBlt(
                compatibleDc,
                0,
                0,
                width,
                height,
                screenDc,
                0,
                0,
                Gdi32.TernaryRasterOperations.SRCCOPY | Gdi32.TernaryRasterOperations.CAPTUREBLT);

            using var image = System.Drawing.Image.FromHbitmap(bitmap);
            using var memory = new MemoryStream();
            image.Save(memory, System.Drawing.Imaging.ImageFormat.Jpeg);
            memory.Position = 0;

            var randomAccessStream = new InMemoryRandomAccessStream();
            randomAccessStream.WriteAsync(memory.GetWindowsRuntimeBuffer()).AsTask().Wait();
            randomAccessStream.Seek(0);

            Gdi32.SelectObject(compatibleDc, old);
            Gdi32.DeleteObject(bitmap);
            Gdi32.DeleteDC(compatibleDc);
            return Task.FromResult<IRandomAccessStream?>(randomAccessStream);
        }
        finally
        {
            Gdi32.ReleaseDC(IntPtr.Zero, screenDc);
        }
    }
}
