# Mangrove Classification Using Machine Learning 

Repository for storing general tools and documentation for the Mangrove Monitoring Project. This includes gisutils and the documentation site for all of our development documentation. 

# Documentaion Site: 
https://ucsd-e4e.github.io/mangrove/


## Using and editing the documentation site

Adding to the documentation site requires simple command line and markdown skills to edit and deploy changes to the site

### Markdown 

For a nice cheatsheet for markdown syntax, check out the cheatsheet linked below:

https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#links

*VSCode is reccomended due to markdown suupport out of the box and a lot of different extensiuons to make editing documents easy*

https://code.visualstudio.com/docs/languages/markdown


### MKDocs 

>MkDocs is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file. Start by reading the introduction below, then check the User Guide for more info." 

https://www.mkdocs.org/

To install mkdocs, assuming you have pip and python prebuilt:

```console
pip install --upgrade pip
```   
```console
pip install mkdocs
```   

Next, `cd` into the mangrove folder which should contain docs, Tools, etc., and use the following to view the site locally

```console
mkdocs serve
```   

Here, you can view the website and changes that you make to the markdown documents within the `/docs` folder.

In order to deploy your changes to the website, from the same directory that you used mkdocs serve with

```console 
mkdocs gh-deploy
```

This will then lead you to a log in prompt where you can then enter your github login info to authenticate and push changes made to the website to the `gh-pages` branch. Keep in mind, this won't deploy your changes to the `master` branch, so make sure to also push your changes of the markdown documents to the master branch as well!
