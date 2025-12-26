from .overview_tab import render as render_overview
from .visualization_tab import render as render_visualization
from .data_tab import render as render_data
from .analysis_tab import render as render_analysis
from .categorical_tab import render as render_categorical

__all__ = ['render_overview', 'render_visualization', 'render_data', 'render_analysis', 'render_categorical']