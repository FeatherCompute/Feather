#include "feather_window_host.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#else
#define GLFW_EXPOSE_NATIVE_X11
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#ifdef None
#undef None
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <variant>

namespace Feather {

namespace {

void trace_window_event(const char* event) {
    const auto* trace = std::getenv("FEATHER_GRAPHICS_TRACE");
    if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
        std::cerr << "[feather window] " << event << '\n';
    }
}

Window::ModifierFlags convert_modifiers(int modifiers) noexcept {
    using Window::ModifierFlags;
    auto result = ModifierFlags::None;
    if ((modifiers & GLFW_MOD_SHIFT) != 0)
        result = result | ModifierFlags::Shift;
    if ((modifiers & GLFW_MOD_CONTROL) != 0)
        result = result | ModifierFlags::Ctrl;
    if ((modifiers & GLFW_MOD_ALT) != 0)
        result = result | ModifierFlags::Alt;
    if ((modifiers & GLFW_MOD_SUPER) != 0)
        result = result | ModifierFlags::Super;
    if ((modifiers & GLFW_MOD_CAPS_LOCK) != 0)
        result = result | ModifierFlags::CapsLock;
    if ((modifiers & GLFW_MOD_NUM_LOCK) != 0)
        result = result | ModifierFlags::NumLock;
    return result;
}

std::mutex glfw_mutex;
size_t glfw_users = 0u;

void acquire_glfw() {
    std::lock_guard lock{glfw_mutex};
    if (glfw_users == 0u && glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("Failed to initialize GLFW for Luisa presentation");
    }
    ++glfw_users;
}

void release_glfw() noexcept {
    std::lock_guard lock{glfw_mutex};
    if (glfw_users != 0u && --glfw_users == 0u) {
        glfwTerminate();
    }
}

class GlfwWindow {
  private:
    GLFWwindow* window_ = nullptr;
    std::queue<Window::Event> events_;
    uint32_t width_ = 0u;
    uint32_t height_ = 0u;
    int32_t mouse_x_ = 0;
    int32_t mouse_y_ = 0;
    float scroll_x_ = 0.0f;
    float scroll_y_ = 0.0f;
    bool open_ = false;
    bool vsync_ = true;

  public:
    explicit GlfwWindow(const Window::Config& config)
        : width_{config.width}, height_{config.height}, vsync_{config.vsync} {
        acquire_glfw();
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_VISIBLE, config.visible ? GLFW_TRUE : GLFW_FALSE);
#if defined(__APPLE__)
        glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, config.high_dpi ? GLFW_TRUE : GLFW_FALSE);
#endif
        window_ = glfwCreateWindow(static_cast<int>(config.width), static_cast<int>(config.height),
                                   config.title.c_str(), nullptr, nullptr);
        if (window_ == nullptr) {
            release_glfw();
            throw std::runtime_error("Failed to create GLFW window for Luisa presentation");
        }
        if (config.center_on_create) {
            if (auto* monitor = glfwGetPrimaryMonitor(); monitor != nullptr) {
                if (const auto* mode = glfwGetVideoMode(monitor); mode != nullptr) {
                    glfwSetWindowPos(window_, std::max(0, (mode->width - static_cast<int>(config.width)) / 2),
                                     std::max(0, (mode->height - static_cast<int>(config.height)) / 2));
                }
            }
        }
        glfwSetWindowUserPointer(window_, this);
        glfwSetWindowSizeCallback(window_, [](GLFWwindow* window, int width, int height) {
            auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
            self->width_ = static_cast<uint32_t>(std::max(width, 0));
            self->height_ = static_cast<uint32_t>(std::max(height, 0));
            self->events_.emplace(Window::ResizeEvent{self->width_, self->height_});
        });
        glfwSetWindowCloseCallback(window_, [](GLFWwindow* window) {
            auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
            trace_window_event("native close callback");
            self->open_ = false;
            self->events_.emplace(Window::CloseEvent{});
        });
        glfwSetWindowFocusCallback(window_, [](GLFWwindow* window, int focused) {
            auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
            self->events_.emplace(Window::FocusEvent{focused == GLFW_TRUE});
        });
        glfwSetKeyCallback(window_, [](GLFWwindow* window, int key, int, int action, int modifiers) {
            if (action == GLFW_REPEAT)
                return;
            auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
            self->events_.emplace(Window::KeyEvent{static_cast<Window::Key>(key), action == GLFW_PRESS,
                                                        convert_modifiers(modifiers)});
        });
        glfwSetCharCallback(window_, [](GLFWwindow* window, unsigned int codepoint) {
            auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
            self->events_.emplace(Window::CharInputEvent{codepoint});
        });
        glfwSetMouseButtonCallback(window_, [](GLFWwindow* window, int button, int action, int modifiers) {
            if (action == GLFW_REPEAT)
                return;
            auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
            self->events_.emplace(Window::MouseButtonEvent{static_cast<Window::MouseButton>(button),
                                                                action == GLFW_PRESS, self->mouse_x_, self->mouse_y_,
                                                                convert_modifiers(modifiers)});
        });
        glfwSetCursorPosCallback(window_, [](GLFWwindow* window, double x, double y) {
            auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
            const auto next_x = static_cast<int32_t>(std::lround(x));
            const auto next_y = static_cast<int32_t>(std::lround(y));
            self->events_.emplace(
                Window::MouseMoveEvent{next_x, next_y, next_x - self->mouse_x_, next_y - self->mouse_y_});
            self->mouse_x_ = next_x;
            self->mouse_y_ = next_y;
        });
        glfwSetScrollCallback(window_, [](GLFWwindow* window, double x, double y) {
            auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
            self->scroll_x_ = static_cast<float>(x);
            self->scroll_y_ = static_cast<float>(y);
            self->events_.emplace(Window::MouseScrollEvent{self->scroll_x_, self->scroll_y_});
        });
        double mouse_x = 0.0;
        double mouse_y = 0.0;
        glfwGetCursorPos(window_, &mouse_x, &mouse_y);
        mouse_x_ = static_cast<int32_t>(std::lround(mouse_x));
        mouse_y_ = static_cast<int32_t>(std::lround(mouse_y));
        open_ = true;
    }

    ~GlfwWindow() {
        if (window_ != nullptr) {
            glfwDestroyWindow(window_);
            release_glfw();
        }
    }

    [[nodiscard]] bool is_open() const noexcept {
        return open_ && window_ != nullptr && glfwWindowShouldClose(window_) == GLFW_FALSE;
    }
    void close() {
        open_ = false;
        if (window_ != nullptr)
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
    [[nodiscard]] uint32_t width() const noexcept {
        return width_;
    }
    [[nodiscard]] uint32_t height() const noexcept {
        return height_;
    }
    void set_title(const std::string& title) {
        glfwSetWindowTitle(window_, title.c_str());
    }
    void set_vsync(bool enabled) noexcept {
        vsync_ = enabled;
    }
    [[nodiscard]] bool vsync() const noexcept {
        return vsync_;
    }
    void poll_events() {
        scroll_x_ = 0.0f;
        scroll_y_ = 0.0f;
        glfwPollEvents();
        if (glfwWindowShouldClose(window_) == GLFW_TRUE)
            open_ = false;
    }
    void wait_events() {
        scroll_x_ = 0.0f;
        scroll_y_ = 0.0f;
        glfwWaitEvents();
        if (glfwWindowShouldClose(window_) == GLFW_TRUE)
            open_ = false;
    }
    bool poll_event(Window::Event& event) {
        if (events_.empty())
            return false;
        event = events_.front();
        events_.pop();
        return true;
    }
    [[nodiscard]] bool is_key_down(Window::Key key) const {
        return window_ != nullptr && static_cast<int32_t>(key) >= 0 &&
               glfwGetKey(window_, static_cast<int>(key)) == GLFW_PRESS;
    }
    [[nodiscard]] bool is_mouse_down(Window::MouseButton button) const {
        return window_ != nullptr && glfwGetMouseButton(window_, static_cast<int>(button)) == GLFW_PRESS;
    }
    [[nodiscard]] std::pair<int32_t, int32_t> mouse_position() const noexcept {
        return {mouse_x_, mouse_y_};
    }
    [[nodiscard]] std::pair<float, float> mouse_scroll() const noexcept {
        return {scroll_x_, scroll_y_};
    }
    [[nodiscard]] uint64_t native_display() const noexcept {
#if defined(_WIN32) || defined(__APPLE__)
        return 0u;
#else
        return reinterpret_cast<uint64_t>(glfwGetX11Display());
#endif
    }
    [[nodiscard]] uint64_t native_window() const noexcept {
#if defined(_WIN32)
        return reinterpret_cast<uint64_t>(glfwGetWin32Window(window_));
#elif defined(__APPLE__)
        return reinterpret_cast<uint64_t>(glfwGetCocoaWindow(window_));
#else
        return static_cast<uint64_t>(glfwGetX11Window(window_));
#endif
    }
};

} // namespace

class WindowHost::Impl {
  public:
    explicit Impl(const Window::Config& config) : config{config}, window{std::make_unique<GlfwWindow>(config)} {}
    Window::Config config;
    std::unique_ptr<GlfwWindow> window;
};

WindowHost::WindowHost(const Window::Config& config) : impl_{std::make_unique<Impl>(config)} {}
WindowHost::~WindowHost() = default;
bool WindowHost::IsOpen() const noexcept {
    return impl_->window->is_open();
}
void WindowHost::Close() {
    impl_->window->close();
}
uint32_t WindowHost::Width() const noexcept {
    return impl_->window->width();
}
uint32_t WindowHost::Height() const noexcept {
    return impl_->window->height();
}
void WindowHost::SetTitle(const std::string& title) {
    impl_->window->set_title(title);
}
void WindowHost::SetVSync(bool enabled) {
    impl_->config.vsync = enabled;
    impl_->window->set_vsync(enabled);
}
void WindowHost::PollEvents() {
    impl_->window->poll_events();
}
void WindowHost::WaitEvents() {
    impl_->window->wait_events();
}
bool WindowHost::PollEvent(Window::Event& event) {
    return impl_->window->poll_event(event);
}
bool WindowHost::IsKeyDown(Window::Key key) const {
    return impl_->window->is_key_down(key);
}
bool WindowHost::IsMouseDown(Window::MouseButton button) const {
    return impl_->window->is_mouse_down(button);
}
std::pair<int32_t, int32_t> WindowHost::MousePosition() const noexcept {
    return impl_->window->mouse_position();
}
std::pair<float, float> WindowHost::MouseScroll() const noexcept {
    return impl_->window->mouse_scroll();
}
uint64_t WindowHost::NativeDisplay() const noexcept {
    return impl_->window->native_display();
}
uint64_t WindowHost::NativeWindow() const noexcept {
    return impl_->window->native_window();
}
bool WindowHost::VSync() const noexcept {
    return impl_->window->vsync();
}

} // namespace Feather
