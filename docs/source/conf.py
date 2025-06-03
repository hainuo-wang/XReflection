# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'XReflection'
copyright = '2025, Peiyuan He'
author = 'Peiyuan He'
release = '0.1.0_beta'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

pygments_style = 'sphinx'


# -- Options for HTML output

# html_theme = 'sphinx_rtd_theme'
html_theme = 'conestack'

# 自定义logo配置
html_logo = '../_static/XReflection_logo.png'

# 静态文件路径
html_static_path = ['../_static']

# 强制设置项目标题
html_title = 'XReflection Documentation'
html_short_title = 'XReflection'

# 主题选项配置（移除不支持的选项）
html_theme_options = {
    # conestack主题可能不支持logo_only选项，所以注释掉
    # 'logo_only': False,
}

# 如果需要自定义CSS样式
html_css_files = [
    'custom.css',  # 可选：用于进一步自定义logo样式
]

# -- Options for EPUB output
epub_show_urls = 'footnote'