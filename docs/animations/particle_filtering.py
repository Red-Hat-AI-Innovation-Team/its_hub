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

        prompt = labeled_box("Prompt", color=ACCENT_BLUE, width=1.6, height=0.55)
        prompt.move_to(LEFT * 6 + DOWN * 0.3)
        self.play(FadeIn(prompt), run_time=0.8)

        prm_label = labeled_box("PRM", color=ACCENT_ORANGE, width=1.2, height=0.5, font_size=18)
        prm_label.move_to(UP * 3.2)
        self.play(FadeIn(prm_label), run_time=0.6)

        step_x = [-3.5, -0.5, 2.5]
        for i, x in enumerate(step_x):
            sl = Text(f"Step {i+1}", font_size=14)
            sl.set_color(INACTIVE_GRAY)
            sl.move_to(RIGHT * x + UP * 2.5)
            self.add(sl)

        self.wait(0.5)

        n_particles = 5
        particle_y = np.linspace(1.5, -2.5, n_particles)

        particles = VGroup()
        for idx, y in enumerate(particle_y):
            p = labeled_box(f"P{idx+1}", color=PARTICLE_COLORS[idx], width=1.0, height=0.4, font_size=14)
            p.move_to(LEFT * 4.8 + UP * y)
            particles.add(p)

        init_arrows = VGroup(*[
            thin_arrow(prompt.get_right(), p.get_left()) for p in particles
        ])

        self.play(
            LaggedStart(*[ShowCreation(a) for a in init_arrows], lag_ratio=0.05),
            LaggedStart(*[FadeIn(p) for p in particles], lag_ratio=0.05),
            run_time=1.0,
        )
        self.wait(0.5)

        weights_per_step = [
            [0.3, 0.7, 0.5, 0.9, 0.2],
            [0.4, 0.8, 0.3, 0.6, 0.8],
            [0.2, 0.9, 0.5, 0.7, 0.6],
        ]
        # Multinomial resampling: each slot drawn from full distribution
        # These represent the source particle index for each of the 5 slots
        resample_draws = [
            [3, 1, 3, 1, 3],   # weights [.3,.7,.5,.9,.2] → P4(0.9) drawn 3x, P2(0.7) drawn 2x
            [1, 4, 1, 4, 3],   # weights [.4,.8,.3,.6,.8] → P2,P5(0.8) drawn 2x each, P4(0.6) 1x
            None,               # no resample on last step
        ]

        current_particles = list(particles)
        current_colors = list(PARTICLE_COLORS)

        for step in range(3):
            x_target = step_x[step]
            weights = weights_per_step[step]

            # Generate new particles at this step
            new_particles = VGroup()
            move_arrows = VGroup()
            for i, p in enumerate(current_particles):
                new_p = labeled_box(f"P{i+1}", color=current_colors[i], width=1.0, height=0.4, font_size=14)
                new_p.move_to(RIGHT * x_target + UP * particle_y[i])
                new_particles.add(new_p)
                arr = thin_arrow(p.get_right(), new_p.get_left())
                move_arrows.add(arr)

            self.play(
                LaggedStart(*[ShowCreation(a) for a in move_arrows], lag_ratio=0.04),
                LaggedStart(*[FadeIn(p) for p in new_particles], lag_ratio=0.04),
                run_time=0.8,
            )

            # PRM scoring arrows + score labels
            prm_arrows = VGroup(*[
                thin_arrow(prm_label.get_bottom(), p.get_top(), color=ACCENT_ORANGE)
                for p in new_particles
            ])
            self.play(
                LaggedStart(*[ShowCreation(a) for a in prm_arrows], lag_ratio=0.08),
                run_time=0.8,
            )

            score_labels = VGroup()
            best_w = max(weights)
            for i, (p, w) in enumerate(zip(new_particles, weights)):
                sl = Text(f"{w}", font_size=14)
                sl.set_color(ACCENT_GREEN if w == best_w else TEXT_COLOR)
                sl.next_to(p, RIGHT, buff=0.15)
                score_labels.add(sl)

            self.play(
                LaggedStart(*[FadeIn(s) for s in score_labels], lag_ratio=0.08),
                run_time=0.8,
            )
            self.wait(0.8)

            # Fade PRM arrows
            self.play(*[FadeOut(a) for a in prm_arrows], run_time=0.3)

            # Resample with restitution from the full distribution
            if resample_draws[step] is not None:
                draws = resample_draws[step]

                # Show "Resample" label briefly
                resample_text = Text("resample", font_size=16)
                resample_text.set_color(ACCENT_ORANGE)
                resample_text.move_to(RIGHT * x_target + DOWN * 3.2)
                self.play(FadeIn(resample_text), run_time=0.3)

                # Fade ALL current particles (every slot gets redrawn)
                self.play(
                    *[p.animate.set_opacity(0.15) for p in new_particles],
                    *[s.animate.set_opacity(0.15) for s in score_labels],
                    run_time=0.5,
                )

                # Draw all 5 new particles from the distribution simultaneously
                # Each takes the color/label of its source
                resampled = []
                new_colors = []
                draw_anims = []
                for i in range(n_particles):
                    src = draws[i]
                    src_color = current_colors[src]
                    new_colors.append(src_color)
                    drawn_p = labeled_box(f"P{src+1}", color=src_color, width=1.0, height=0.4, font_size=14)
                    drawn_p.move_to(new_particles[src].get_center())
                    drawn_p.set_opacity(0)
                    self.add(drawn_p)
                    draw_anims.append(
                        drawn_p.animate.move_to(new_particles[i].get_center()).set_opacity(1)
                    )
                    resampled.append(drawn_p)

                self.play(*draw_anims, run_time=1.0)
                self.play(FadeOut(resample_text), run_time=0.2)

                current_particles = resampled
                current_colors = new_colors
            else:
                current_particles = list(new_particles)

            self.wait(0.3)

        # Select best particle
        self.wait(0.5)
        best_idx = weights_per_step[2].index(max(weights_per_step[2]))
        best_particle = current_particles[best_idx]

        winner = labeled_box("✓ Best", color=ACCENT_GREEN, width=1.4, height=0.45, font_size=16)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.move_to(RIGHT * 5 + UP * particle_y[best_idx])

        winner_arrow = thin_arrow(best_particle.get_right(), winner.get_left(), color=ACCENT_GREEN)
        self.play(ShowCreation(winner_arrow), FadeIn(winner), run_time=1.0)
        self.wait(4.0)
