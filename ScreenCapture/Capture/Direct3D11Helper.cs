using System.Runtime.InteropServices;
using Windows.Graphics.DirectX.Direct3D11;

namespace ScreenCapture.Capture;

internal static class Direct3D11Helper
{
    private const string D3D11 = "d3d11.dll";

    public static IDirect3DDevice CreateDevice()
    {
        var devicePtr = CreateD3DDevice();
        var dxgiGuid = typeof(IDXGIDevice).GUID;
        Marshal.ThrowExceptionForHR(Marshal.QueryInterface(devicePtr, ref dxgiGuid, out var dxgiPtr));

        try
        {
            var dxgiDevice = Marshal.GetObjectForIUnknown(dxgiPtr);
            Marshal.ThrowExceptionForHR(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice!, out var graphicsDevice));
            return graphicsDevice;
        }
        finally
        {
            Marshal.Release(dxgiPtr);
            Marshal.Release(devicePtr);
        }
    }

    private static IntPtr CreateD3DDevice()
    {
        var flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        var featureLevels = new[]
        {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        };

        Marshal.ThrowExceptionForHR(D3D11CreateDevice(
            IntPtr.Zero,
            D3D_DRIVER_TYPE_HARDWARE,
            IntPtr.Zero,
            flags,
            featureLevels,
            featureLevels.Length,
            D3D11_SDK_VERSION,
            out var device,
            out _,
            out var context));

        if (context != IntPtr.Zero)
        {
            Marshal.Release(context);
        }

        return device;
    }

    [DllImport(D3D11, ExactSpelling = true)]
    private static extern int D3D11CreateDevice(
        IntPtr pAdapter,
        int driverType,
        IntPtr software,
        int flags,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5)] uint[] pFeatureLevels,
        int featureLevels,
        uint sdkVersion,
        out IntPtr ppDevice,
        out IntPtr pFeatureLevel,
        out IntPtr ppImmediateContext);

    [DllImport(D3D11, EntryPoint = "CreateDirect3D11DeviceFromDXGIDevice", SetLastError = true)]
    private static extern int CreateDirect3D11DeviceFromDXGIDevice(
        [MarshalAs(UnmanagedType.IUnknown)] object dxgiDevice,
        out IDirect3DDevice graphicsDevice);

    [ComImport, Guid("77db970f-6276-48ba-ba28-070143b4392c"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIDevice
    {
    }

    private const int D3D_DRIVER_TYPE_HARDWARE = 1;
    private const int D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20;
    private const uint D3D11_SDK_VERSION = 7;

    private const uint D3D_FEATURE_LEVEL_10_0 = 0xb000;
    private const uint D3D_FEATURE_LEVEL_10_1 = 0xb100;
    private const uint D3D_FEATURE_LEVEL_11_0 = 0xb0000;
    private const uint D3D_FEATURE_LEVEL_11_1 = 0xb1000;
}
