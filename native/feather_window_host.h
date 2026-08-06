#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>

namespace Feather {

namespace Window {

struct Config {
    uint32_t width = 1280u;
    uint32_t height = 720u;
    std::string title = "Feather";
    bool resizable = true;
    bool visible = true;
    bool vsync = true;
    bool high_dpi = true;
    bool center_on_create = true;
};

enum class Key : int32_t {};
enum class MouseButton : uint8_t {};

enum class ModifierFlags : uint32_t {
    None = 0u,
    Shift = 1u << 0u,
    Ctrl = 1u << 1u,
    Alt = 1u << 2u,
    Super = 1u << 3u,
    CapsLock = 1u << 4u,
    NumLock = 1u << 5u,
};

constexpr ModifierFlags operator|(ModifierFlags left, ModifierFlags right) noexcept {
    return static_cast<ModifierFlags>(static_cast<uint32_t>(left) | static_cast<uint32_t>(right));
}

struct ResizeEvent { uint32_t width; uint32_t height; };
struct CloseEvent {};
struct KeyEvent { Key key; bool pressed; ModifierFlags modifiers; };
struct CharInputEvent { uint32_t codepoint; };
struct MouseButtonEvent { MouseButton button; bool pressed; int32_t x; int32_t y; ModifierFlags modifiers; };
struct MouseMoveEvent { int32_t x; int32_t y; int32_t dx; int32_t dy; };
struct MouseScrollEvent { float dx; float dy; };
struct FocusEvent { bool focused; };
using Event = std::variant<ResizeEvent, CloseEvent, KeyEvent, CharInputEvent, MouseButtonEvent,
                           MouseMoveEvent, MouseScrollEvent, FocusEvent>;

} // namespace Window

class WindowHost {
  public:
    explicit WindowHost(const Window::Config& config);
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
    bool PollEvent(Window::Event& event);
    [[nodiscard]] bool IsKeyDown(Window::Key key) const;
    [[nodiscard]] bool IsMouseDown(Window::MouseButton button) const;
    [[nodiscard]] std::pair<int32_t, int32_t> MousePosition() const noexcept;
    [[nodiscard]] std::pair<float, float> MouseScroll() const noexcept;
    [[nodiscard]] uint64_t NativeDisplay() const noexcept;
    [[nodiscard]] uint64_t NativeWindow() const noexcept;
    [[nodiscard]] bool VSync() const noexcept;
  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Feather
