"""Small responsive loading window for blocking desktop tasks."""

import os
import queue
import threading


def run_with_square_progress(task, *, title, label):
    """Run ``task(progress_callback)`` while keeping a Pygame window responsive."""
    result_queue = queue.Queue()
    progress_queue = queue.Queue()

    def report_progress(completed, total):
        progress_queue.put((int(completed), max(0, int(total))))

    def worker():
        try:
            result_queue.put((True, task(report_progress)))
        except BaseException as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=worker, name="solver-merged-generation", daemon=True)
    thread.start()

    try:
        import pygame
    except ImportError:
        thread.join()
        success, value = result_queue.get()
        if success:
            return value
        raise value

    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.init()
    screen = pygame.display.set_mode((620, 230))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("Arial", 27, bold=True)
    label_font = pygame.font.SysFont("Arial", 19)
    completed = 0
    total = 0
    square_count = 14

    try:
        while thread.is_alive() or result_queue.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass
            while True:
                try:
                    completed, total = progress_queue.get_nowait()
                except queue.Empty:
                    break

            ratio = 0.0 if total <= 0 else min(1.0, completed / total)
            active_count = int(ratio * square_count)
            if completed and active_count == 0:
                active_count = 1
            pulse_index = (pygame.time.get_ticks() // 110) % square_count

            screen.fill((12, 16, 22))
            heading = title_font.render(title, True, (242, 247, 252))
            screen.blit(heading, heading.get_rect(center=(310, 54)))
            detail = label_font.render(label, True, (160, 205, 235))
            screen.blit(detail, detail.get_rect(center=(310, 94)))

            square_size = 24
            gap = 10
            bar_width = square_count * square_size + (square_count - 1) * gap
            start_x = (screen.get_width() - bar_width) // 2
            for index in range(square_count):
                rect = pygame.Rect(start_x + index * (square_size + gap), 128, square_size, square_size)
                is_active = index < active_count or (completed == 0 and index == pulse_index)
                color = (80, 190, 245) if is_active else (36, 49, 62)
                pygame.draw.rect(screen, color, rect, border_radius=4)
                pygame.draw.rect(screen, (105, 205, 255), rect, 1, border_radius=4)

            percent = 0 if total <= 0 else round(ratio * 100)
            counter = label_font.render(f"Progression : {percent}%", True, (210, 222, 235))
            screen.blit(counter, counter.get_rect(center=(310, 188)))
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()

    success, value = result_queue.get()
    if success:
        return value
    raise value
