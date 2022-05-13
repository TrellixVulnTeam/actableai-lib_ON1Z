Developer Contribution
======================

You can join (and Star!) us on on `GitHub <https://github.com/Actable-AI/actableai-app>`__.

Contributing to Actable AI
--------------------------

We welcome all forms of contributions to Actable AI, including and not limited to:

* Code reviewing of patches and PRs.

* Pushing patches.

* Documentation and examples.

* Community participation in forums and issues.

* Code readability and code comments to improve readability.

* Test cases to make the codebase more robust.

* Tutorials, blog posts, talks that promote the project.

**What can I work on?**

We use Github to track issues, feature requests, and bugs.

Setting up your development environment
---------------------------------------

To edit the Actable source code, you’ll want to checkout the repository and also 
build Actable from source. Follow these instructions for building a local copy of 
Actble AI to easily make changes. 

Submitting and merging a contribution
-------------------------------------

There are a couple steps to merge a contribution.

1. First merge the most recent version of master into your development branch.

   .. code:: bash
   
      git remote add upstream https://github.com/Actable-AI/actableai-app.git    
      git pull . upstream/master

2. Make sure all existing tests and linters pass. Run setup_hooks.sh to create a git hook 
that will run the linter before you push your changes.

3. If introducing a new feature or patching a bug, be sure to add new test cases in 
the relevant file.

4. Document the code. Public functions need to be documented, and remember to 
provide an usage example if applicable. See doc/README.md for instructions 
on editing and building public documentation.

5. Address comments on your PR. During the review process you may need to address merge 
conflicts with other changes. To resolve merge conflicts, run git pull . upstream/master 
on your branch (please do not use rebase, as it is less friendly to the GitHub 
review tool. All commits will be squashed on merge.)

6. Reviewers will merge and approve the pull request; be sure to ping them if the 
pull request is getting stale.

PR Review Process
-----------------

For contributors who are in the ``Actable AI`` organization:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- When you first create a PR, add an reviewer to the `assignee` section.
- Assignees will review your PR and add the `@author-action-required` label if further actions are required.
- Address their comments and remove the `@author-action-required` label from the PR.
- Repeat this process until assignees approve your PR.
- Once the PR is approved, the author is in charge of ensuring the PR passes the build. Add the `test-ok` label if the build succeeds.
- Committers will merge the PR once the build is passing.

For contributors who are not in the ``Actable AI`` organization:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Your PRs will have assignees shortly. Assignees of PRs will be actively engaging with contributors to merge the PR.
- Please actively ping assignees after you address your comments!

Testing
-------

Even though we have hooks to run unit tests automatically for each pull request,
we recommend you to run unit tests locally beforehand to reduce reviewersâ€™
burden and speedup review process.

If you are running tests for the first time, you can install the required dependencies with:

.. code-block:: shell

    pip install -r python/requirements.txt
    pytest superset/
    
Code Style
----------

In general, we follow the `Google style guide <https://google.github.io/styleguide/>`__ for code in Python. 
However, it is more important for code to be in a locally consistent style than to strictly follow guidelines. 
Whenever in doubt, follow the local code style of the component.

For Python documentation, we follow a subset of the `Google pydoc format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__. 
The following code snippet demonstrates the canonical Actable AI pydoc formatting:

.. code-block:: python

    def canonical_doc_style(param1: int, param2: str) -> bool:
        """First sentence MUST be inline with the quotes and fit on one line.

        Additional explanatory text can be added in paragraphs such as this one.
        Do not introduce multi-line first sentences.

        Examples:
            >>> # Provide code examples as possible.
            >>> canonical_doc_style(41, "hello")
            True

            >>> # A second example.
            >>> canonical_doc_style(72, "goodbye")
            False

        Args:
            param1: The first parameter. Do not include the types in the
                docstring (they should be defined only in the signature).
                Multi-line parameter docs should be indented by four spaces.
            param2: The second parameter.

        Returns:
            The return value. Do not include types here.
        """


Lint and Formatting
~~~~~~~~~~~~~~~~~~~

We also have tests for code formatting and linting that need to pass before merge.

* For Python formatting, install the `required dependencies <https://github.com/Actable-AI/actableai-app/blob/master/requirements_linters.txt>`_ first with:

.. code-block:: shell

  pip install -r requirements_linters.txt
  
You can run the following locally:

.. code-block:: shell

    ./format.sh

**Other recommendations**:

In Python APIs, consider forcing the use of kwargs instead of positional arguments (with the ``*`` operator). Kwargs are easier to keep backwards compatible than positional arguments, e.g. imagine if you needed to deprecate "opt1" below, it's easier with forced kwargs:

.. code-block:: python

    def foo_bar(file, *, opt1=x, opt2=y)
        pass

For callback APIs, consider adding a ``**kwargs`` placeholder as a "forward compatibility placeholder" in case more args need to be passed to the callback in the future, e.g.:
 
.. code-block:: python

    def tune_user_callback(model, score, **future_kwargs):
        pass

Becoming a Reviewer
-------------------

We identify reviewers from active contributors. Reviewers are individuals who
not only actively contribute to the project and are also willing
to participate in the code review of new contributions.
A pull request to the project has to be reviewed by at least one reviewer in order to be merged.
There is currently no formal process, but active contributors to Actable AI will be
solicited by current reviewers.
