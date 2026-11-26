"""Pin LIBERO offscreen rendering to zero multisample samples.

Default MSAA rendering produced one-level RGB differences on identical state
and geometry. This is an explicit renderer-protocol change, not a tolerance or
postprocessing of observations. Existing default-renderer outcomes stay separate.
"""
def install():
    from robosuite.utils.binding_utils import MjRenderContext
    selected=getattr(MjRenderContext,'_exact_replay_offscreen_samples',None)
    if selected is not None:
        if selected!=0:raise RuntimeError('A different exact-replay renderer is already installed')
        return
    original=MjRenderContext._set_mujoco_context_and_buffers
    def create(self):
        self.model.vis.quality.offsamples=0
        original(self)
        if self.con.offSamples!=0:raise RuntimeError('Offscreen renderer ignored the pinned sample count')
    MjRenderContext._set_mujoco_context_and_buffers=create
    MjRenderContext._exact_replay_offscreen_samples=0
