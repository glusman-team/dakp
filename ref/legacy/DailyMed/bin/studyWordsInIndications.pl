#!/bin/env perl
use strict;

my $file = "extracted/indications.txt";
my $outfile = "results/indication-words.txt";

my($n, $seen) = studyFile($file);


open OUTF, ">$outfile";
foreach my $word (sort {$seen->{$b}<=>$seen->{$a}} keys %$seen) {
        print OUTF join("\t", sprintf("%.4f", $seen->{$word}/$n), $word), "\n";
}
close OUTF;



####
sub studyFile {
	my($file) = @_;
	my($n, %seen);
	if ($file =~ /\.gz$/) {
		open F, "gunzip -c $file |";
	} else {
		open F, $file;
	}
	while (<F>) {
		chomp;
		my($xml, $text) = split /\t/;
		my $words = studyText(lc $text);
		$seen{$_}++ foreach keys %$words;
		$n++;
	}
	close F;
	return $n, \%seen;
}

sub studyText {
	my($string) = @_;
	$string =~ s/\<[^\>]*\>/ /g;
	my %words;
	$words{$_}++ foreach split /\W+/, $string;
	return \%words;
}	
