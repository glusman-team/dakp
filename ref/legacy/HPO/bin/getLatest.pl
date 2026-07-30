#!/bin/env perl
use strict;

my $logfile = "getlatest.log";
my $idx = "releases";

unlink $idx;
`wget https://github.com/obophenotype/human-phenotype-ontology/releases --no-check-certificate` unless -s $idx;
die "failed to get $idx\n" unless -s $idx;

mkdir "data", 0755;
open IDX, $idx;
my $file;
my $url;
my $latest;
while (<IDX>) {
	if (/<a href=\"(.+?\/hp-base.obo)\"/) {
		$url = $1;
		print "https://github.com$url\n";
		doLog("Downloading https://github.com$url");
		`wget https://github.com$url -O data/hp-base.obo`;
	}
}
close IDX;
doLog("Done downloading.");

###
sub doLog {
	my($msg) = @_;
	
	open LOG, ">>$logfile";
	my $now = `date`;
	chomp($now);
	print LOG join("\t", $now, $msg), "\n";
	close LOG;
}

