#!/bin/env perl
use strict;
require "../lib/libSystem.pl";

my($checkForNew, $forceRebuild) = @ARGV;
my $logfile = "getlatest.log";
my $idx = "FPD-QDE-FAERS.html";

unlink $idx if $checkForNew;
`wget https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html --no-check-certificate` unless -s $idx;
die "failed to get $idx\n" unless -s $idx;

mkdir "data", 0755;
open IDX, $idx;
my $file;
my $url;
my $latest;
my $anythingNew = 0;
while (<IDX>) {
	if (/<a href=\"(https:\/\/fis.fda.gov\/content\/Exports)\/(f?aers_ascii_....[Qq]..zip)\">/) {
		$url = $1;
		$file = $2;
		#print "$url/$file\n";
		next if -e "zips/$file";
		doLog("Downloading $url/$file") unless -s "zips/$file";
		`wget -c $url/$file -O zips/$file`;
		`unzip -u -j zips/$file '*.txt' -d data`;
		`unzip -u -j zips/$file '*.TXT' -d data`;
		`gzip -f data/*.txt`;
		`gzip -f data/*.TXT`;
		$anythingNew++;
	}
}
close IDX;
doLog("Done downloading: $anythingNew new");
exit unless $anythingNew || $forceRebuild;

foreach my $file (sort(fulldirlist("data"))) {
	my($part, $q) = $file =~ /(.*?)(\d\dq\d)/i;
	$q ||= 'ALL';
	$part = uc $part;
	if ($file =~ /delete/i) {
		$part = "DELETE";
	}
	my $new = join("", $part, uc $q, ".txt.gz");
	if ($new ne $file) {
		doLog("renaming $file to $new");
		if (-e "data/$new") {
			doLog("will not clobber file, exiting");
			die;
		}
		rename "data/$file", "data/$new";
		if (-e "data/$file" && !-e "data/$new") {
			doLog("failed to rename $file to $new");
			die;
		}
		`chmod a-w data/$new`;
	}
}

# stash previous results
foreach my $file (qw/cases indications drugs/) {
	if (-s "results/$file.txt") {
		rename "results/$file.txt", "results/${file}-old.txt";
	} elsif (-s "results/$file.txt.gz") {
		rename "results/$file.txt.gz", "results/${file}.txt-old.gz";
	}
}

`bin/listCases.pl`;
`../bin/computeTableStats.pl results/cases.txt.gz > results/cases.stats`;
`bin/drug2indi.pl`;
`python3 bin/findIndicationTerms.py`;


###
sub doLog {
	my($msg) = @_;
	
	open LOG, ">>$logfile";
	my $now = `date`;
	chomp($now);
	print LOG join("\t", $now, $msg), "\n";
	close LOG;
}

