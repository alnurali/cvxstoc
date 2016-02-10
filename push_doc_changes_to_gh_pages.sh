#!/bin/bash

echo "This script presumes that you ran 'make html' from the sphinx folder, hit enter to continue."

echo "Copying over latest doc files from the sphinx folder."
cp -r ../sphinx/_build/html/* .
git add --all
git commit -m "doc updates."

echo "Pushing to github."
git push origin gh-pages

echo "Done!"
