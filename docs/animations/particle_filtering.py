from manimlib import *
from common import *
import numpy as np

PARTICLE_COLORS = ["#4A90D9", "#E06C75", "#61AFEF", "#C678DD", "#E5C07B"]

class ParticleFilteringScene(Scene):
    def construct(self):
        title = Text("Particle Filtering", font_size=30)
        title.set_color(TEXT_COLOR)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), run_time=0.8)

        prm_label = labeled_box("PRM", color=ACCENT_ORANGE, width=1.2, height=0.5, font_size=18)
        prm_label.move_to(UP * 2.8 + RIGHT * 5)
        self.play(FadeIn(prm_label), run_time=0.6)

        prompt = labeled_box("Prompt", color=ACCENT_BLUE, width=1.6, height=0.55)
        prompt.move_to(LEFT * 5.5 + DOWN * 0.5)
        self.play(FadeIn(prompt), run_time=0.8)

        step_x = [-2.5, 0.5, 3.5]
        for i, x in enumerate(step_x):
            sl = Text(f"Step {i+1}", font_size=16)
            sl.set_color(INACTIVE_GRAY)
            sl.move_to(RIGHT * x + UP * 2.2)
            self.add(sl)

        n_particles = 5
        particle_y = np.linspace(1.2, -2.2, n_particles)

        particles = VGroup()
        for idx, y in enumerate(particle_y):
            p = Dot(radius=0.12, fill_color=PARTICLE_COLORS[idx], fill_opacity=0.8)
            p.move_to(LEFT * 4.0 + UP * y)
            particles.add(p)

        init_arrows = VGroup(*[
            thin_arrow(prompt.get_right(), p.get_left()) for p in particles
        ])

        self.play(
            LaggedStart(*[ShowCreation(a) for a in init_arrows], lag_ratio=0.05),
            LaggedStart(*[FadeIn(p) for p in particles], lag_ratio=0.05),
            run_time=1.2,
        )

        weights_per_step = [
            [0.3, 0.7, 0.5, 0.9, 0.2],
            [0.4, 0.8, 0.3, 0.6, 0.8],
            [0.2, 0.9, 0.5, 0.7, 0.6],
        ]
        resample_from = [
            [1, 3, 2, 3, 1],
            [1, 1, 4, 3, 4],
            None,
        ]

        current_particles = list(particles)
        current_colors = list(PARTICLE_COLORS)

        for step in range(3):
            x_target = step_x[step]
            weights = weights_per_step[step]

            new_particles = VGroup()
            move_arrows = VGroup()
            for i, p in enumerate(current_particles):
                new_p = Dot(radius=0.12, fill_color=current_colors[i], fill_opacity=0.8)
                new_p.move_to(RIGHT * x_target + UP * particle_y[i])
                new_particles.add(new_p)
                arr = thin_arrow(p.get_right(), new_p.get_left())
                move_arrows.add(arr)

            self.play(
                LaggedStart(*[ShowCreation(a) for a in move_arrows], lag_ratio=0.04),
                LaggedStart(*[FadeIn(p) for p in new_particles], lag_ratio=0.04),
                run_time=1.0,
            )

            weight_anims = []
            for i, (p, w) in enumerate(zip(new_particles, weights)):
                scale = 0.5 + w * 1.5
                opacity = 0.3 + w * 0.7
                weight_anims.append(p.animate.scale(scale).set_opacity(opacity))

            self.play(*weight_anims, run_time=0.8)

            if resample_from[step] is not None:
                sources = resample_from[step]
                surviving_sources = set(sources)
                eliminated = [i for i in range(n_particles) if i not in surviving_sources]

                fade_anims = [
                    new_particles[i].animate.set_opacity(0.1).set_fill(INACTIVE_GRAY)
                    for i in eliminated
                ]
                if fade_anims:
                    self.play(*fade_anims, run_time=0.6)

                dup_anims = []
                resampled = list(new_particles)
                new_colors = list(current_colors)
                for i in range(n_particles):
                    src = sources[i]
                    if src != i:
                        copy_p = Dot(radius=0.12, fill_color=current_colors[src], fill_opacity=0.8)
                        copy_p.move_to(new_particles[src].get_center())
                        self.add(copy_p)
                        dup_anims.append(
                            copy_p.animate.move_to(new_particles[i].get_center())
                        )
                        resampled[i] = copy_p
                        new_colors[i] = current_colors[src]

                if dup_anims:
                    self.play(*dup_anims, run_time=0.8)

                current_particles = resampled
                current_colors = new_colors
            else:
                current_particles = list(new_particles)

        best_idx = weights_per_step[2].index(max(weights_per_step[2]))
        best_particle = current_particles[best_idx]

        highlight = Circle(radius=0.3, stroke_color=ACCENT_GREEN, stroke_width=3)
        highlight.move_to(best_particle.get_center())

        winner = labeled_box("✓ Best", color=ACCENT_GREEN, width=1.4, height=0.45, font_size=16)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.next_to(highlight, RIGHT, buff=0.3)

        self.play(ShowCreation(highlight), FadeIn(winner), run_time=1.0)
        self.wait(3.0)
