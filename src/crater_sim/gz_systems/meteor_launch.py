"""gz-sim Python system attached to every spawned meteor.

Gazebo SDF has no "initial velocity". Setting a velocity *command* does not work
either: the physics system keeps re-applying the (zeroed) command every step and
pins the body in place. Instead this system applies a single-step impulse,
    F = m * v0 / dt,
on the first simulation step. After that the meteor is a free body under lunar
gravity (no atmosphere, no drag): a true ballistic arc, with natural collisions.

    <plugin filename="gz-sim-python-system-loader-system" name="gz::sim::systems::PythonSystemLoader">
      <module_name>meteor_launch</module_name>
      <mass>50</mass> <vx>..</vx> <vy>..</vy> <vz>..</vz>
    </plugin>
"""
from gz.math7 import Vector3d
from gz.sim8 import Link, Model


class MeteorLaunch:
    def configure(self, entity, sdf, ecm, event_mgr):
        self.model = Model(entity)
        self.mass = sdf.get_double('mass')
        self.v = Vector3d(sdf.get_double('vx'), sdf.get_double('vy'), sdf.get_double('vz'))
        self.done = False

    def pre_update(self, info, ecm):
        if info.paused or self.done:
            return
        dt = info.dt.total_seconds()
        if dt <= 0:
            return
        Link(self.model.canonical_link(ecm)).add_world_force(ecm, self.v * (self.mass / dt))
        self.done = True


def get_system():
    return MeteorLaunch()
