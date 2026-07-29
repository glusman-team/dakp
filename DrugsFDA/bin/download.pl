#!/usr/bin/env perl
$|=1;
use strict;

mkdir "data", 0755;
mkdir "data/tmp", 0755;

# source: https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files
`wget https://www.fda.gov/media/89850/download?attachment -O new.zip` unless -e "new.zip";
`unzip -u -d data/tmp new.zip`;

