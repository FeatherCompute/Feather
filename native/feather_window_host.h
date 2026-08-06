#pragma once

#include <Window/AppWindow.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace Feather {

class WindowHost {
  public:
    WindowHost(const GPU::Window::WindowConfig& config, bool native_presentation);
    ~WindowHost();

    WindowHost(const WindowHost&) = delete;
    WindowHost& operator=(const WindowHost&) = delete;

    [[nodiscard]] bool IsOpen() const noexcept;
    void Close();
    [[nodiscard]] uint32_t Width() const noexcept;
    [[nodiscard]] uint32_t Height() const noexcept;
    void SetTitle(const std::string& title);
    void SetVSync(bool enabled);
    void PollEvents();
    void WaitEvents();
    bool PollEvent(GPU::Window::WindowEvent& event);
    [[nodiscard]] bool IsKeyDown(GPU::Window::Key key) const;
    [[nodiscard]] bool IsMouseDown(GPU::Window::MouseButton button) const;
    [[nodiscard]] std::pair<int32_t, int32_t> MousePosition() const noexcept;
    [[nodiscard]] std::pair<float, float> MouseScroll() const noexcept;
    void Present(const uint32_t* pixels, uint32_t width, uint32_t height);

    [[nodiscard]] bool SupportsNativePresentation() const noexcept;
    [[nodiscard]] uint64_t NativeDisplay() const noexcept;
    [[nodiscard]] uint64_t NativeWindow() const noexcept;
    [[nodiscard]] bool VSync() const noexcept;
    [[nodiscard]] GPU::Window::AppWindow* EasyGpuWindow() noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Feather
