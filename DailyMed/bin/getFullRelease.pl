#!/bin/env perl
use strict;

my $rebuild = shift @ARGV;
my $ddir = "/15TB_1/users/gglusman/DailyMed/downloads";
my $prev = "prev";
my $logfile = "extract.log";
my $index = "index.html";
my $zipsdir = "zips";
my $xmlsdir = "xmls";
my $resultsdir = "results";
my $extracteddir = "extracted";

if (!$rebuild) {
	deleteOlderStuff($ddir);
	stashPrev($ddir, $prev, $index, $logfile, $xmlsdir, $resultsdir, $extracteddir);
	mkdir $zipsdir, 0755;
	mkdir $xmlsdir, 0755;
	mkdir $resultsdir, 0755;
}
getNewDownloadsList($index);
my $files = downloadBigZips($index, $ddir);
my $done = readExtractedLog($logfile);
extractXMLs($files, $ddir, $logfile, $done);

unless (fork) {
	`bin/countCodes.pl > $resultsdir/codes.txt`;
	exit;
}

`python3 bin/parseXML-xtree.py`;
`bin/selectActiveIngredientSingletons.pl`;
`bin/studyWordsInIndications.pl`;
`bin/findTermsInIndications.pl > $resultsdir/terms-in-indications.txt`;
`bin/listActiveIngredientsWithBoxedWarnings.pl > $resultsdir/active_ingredients_with_boxed_warnings.txt`;



#####
sub deleteOlderStuff {
	my($ddir)= @_;
	print "Deleting older downloads\n";
	`rm -r $ddir.prev`;
}

sub stashPrev {
	my($ddir, $prev, @parts) = @_;
	print "Stashing previous downloads\n";
	`mv $ddir $ddir.prev`;
	mkdir $ddir, 0755;
	mkdir $prev, 0755;

	print "Stashing previous content\n";
	foreach my $part (@parts) {
		`mv $part $prev`;
	}
}

sub getNewDownloadsList {
	my($index) = @_;
	print "Getting new downloads list\n";
	`wget https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm --no-check-certificate -O $index` unless -s $index;
	die "failed to get index.html\n" unless -s $index;
}

sub downloadBigZips {
	my($index, $ddir) = @_;
	open IDX, $index;
	while (<IDX>) {
		last if /\<h2\>Full Releases\<\/h2\>/;
	}
	my %files;
	#<a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip">dm_spl_release_human_rx_part1.zip</a>
	while (<IDX>) {
		if (/^\s*\<a href=(\".+?\.zip\")>(.+?\.zip)<\/a>/) {
			my $url = $1;
			my $file = $2;
			$files{$file}++;
			if (-e "$ddir/$file") {
				print "done: $file\n";
			} else {
				print "get: $file\n";
				`wget $url -O $ddir/$file`;
			}
		} elsif (/href/) {
			#print;
		}
	}
	close IDX;
	return \%files;
}

sub readExtractedLog {
	my($logfile) = @_;
	my %done;
	open LOGF, $logfile;
	while (<LOGF>) {
		chomp;
		my($file, $zip, $zipsize, $date, $xmlname, $xmlsize, $xmldate) = split /\t/;
		%{$done{$zip}} = ('file', $file, 'zipsize', $zipsize, 'date', $date, 'xml', $xmlname, 'xmlsize', $xmlsize, 'xmldate', $xmldate);
	}
	close LOGF;
	return \%done;
}

sub extractXMLs {
	my($files, $ddir, $logfile, $done) = @_;
	open LOGF, ">>$logfile";
	foreach my $file (sort keys %$files) {
		print "extracting from $file\n";
		open ZIP, "unzip -l $ddir/$file |";
		while (<ZIP>) { last if /^---/; }
		while (<ZIP>) {
			last if /^---/;
			chomp;
			my($size, $date, $time, $name) = split;
			next if $done->{$name};
			$date .= " $time";
			#print "$_\n";
			`unzip -u -j -d zips $ddir/$file $name`;
			my($shortname) = $name =~ /.+\/(.+)/;
			my $bin = substr($shortname,0,6);
			#print "unzip -l zips/$shortname |\n";
			open IZIP, "unzip -l zips/$shortname |";
			while (<IZIP>) {
				next unless /\.xml/;
				chomp;
				my($xmlsize, $xmldate, $xmltime, $xmlname) = split;
				$xmldate .= " $xmltime";
				`unzip -u -j zips/$shortname $xmlname -d xmls/$bin`;
				`gzip -f xmls/$bin/$xmlname`;
				print LOGF join("\t", $file, $name, $size, $date, $xmlname, $xmlsize, $xmldate), "\n";
			}
			close IZIP;
			unlink "zips/$shortname";
			#exit;
		}
		close ZIP;
	}
	close LOGF;
}



