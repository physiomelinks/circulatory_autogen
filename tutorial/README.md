# Developer documentation for tutorial

## Local preview before a PR

For developers adapting the tutorial locally, check that the formatting renders correctly:

1. Install `Material for MkDocs`:

    ```
    pip install mkdocs-material
    ```

2. Navigate to the `/tutorial` directory and start the live preview server:

    ```
    mkdocs serve
    ```

    The site will be accessible at http://localhost:8000/.
    The server automatically rebuilds on save so you can preview changes as you write.

## Expected outcome

You should be able to browse the tutorial locally and verify links, images, and formatting.

Note: When changes to the `tutorial` directory are merged into the master branch (after a pull request), they are automatically deployed to the website.
