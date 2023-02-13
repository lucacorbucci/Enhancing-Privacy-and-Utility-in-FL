from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.smart import SmartProcessor
from pydoc_markdown.contrib.renderers.docusaurus import DocusaurusRenderer


docs_path = "./docs/docs"

dirlist = ["Models", "Exceptions", "DataSplit", "Utils", "Components"]
for item in dirlist:
    config = PydocMarkdown(
        loaders=[PythonLoader(search_path=[f"pistacchio/{item}"])],
        processors=[
            FilterProcessor(skip_empty_modules=True),
            CrossrefProcessor(),
            SmartProcessor(),
        ],
        renderer=DocusaurusRenderer(
            docs_base_path="./docs/docs/API",
            sidebar_top_level_label=item,
            relative_sidebar_path=f"{item}.json",
            relative_output_path=item,
        ),
    )

    modules = config.load_modules()
    config.process(modules)
    config.render(modules)

dirlist = ["FederatedNode", "P2PCluster", "P2PNode", "SemiP2PNode", "Server"]
for item in dirlist:
    config = PydocMarkdown(
        loaders=[PythonLoader(search_path=[f"pistacchio/Components/{item}"])],
        processors=[
            FilterProcessor(skip_empty_modules=True),
            CrossrefProcessor(),
            SmartProcessor(),
        ],
        renderer=DocusaurusRenderer(
            docs_base_path="./docs/docs/API",
            sidebar_top_level_label=item,
            relative_sidebar_path=f"{item}.json",
            relative_output_path=item,
        ),
    )

    modules = config.load_modules()
    config.process(modules)
    config.render(modules)
