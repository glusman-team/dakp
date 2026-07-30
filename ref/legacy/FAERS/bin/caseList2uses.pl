#!/bin/env perl
use strict;
$|=1;

my($infile, $outfile) = @ARGV;
#$infile ||= "results/cases.txt.gz";
$infile ||= "/ssd2/gglusman/FAERS/cases.txt";
$outfile ||= "| gzip -c > results/uses.txt.gz";

my(%drugcases, %ingcases, %ndas, %drugIngredient, $done, %totalcases);
#open CL, "gunzip -c $infile |";
open CL, $infile;
$_ = <CL>;
my @headers = split /\t/;
my %col;
$col{$headers[$_]} = $_ foreach (0..$#headers);
while (<CL>) {
	chomp;
	my @v = split /\t/;
	my $drugname = $v[$col{'drugname'}];
	#$drugname = cleanText($drugname);
	my $ingredient = $v[$col{'ingredient'}];
	#$ingredient = cleanText($ingredient);
	my $nda = $v[$col{'nda'}];
	$nda = '' if $nda =~ /^(0+|9+)$/; # ignore "dunno" codes like 99, 999999, 000000, 0, etc.
	my $indication = $v[$col{'indication'}];
	next if $indication eq 'product used for unknown indication';
	if ($ingredient) {
		$ingcases{$ingredient}{$indication}++;
		$ndas{$ingredient}{$nda}++ if $nda;
		if ($drugname) {
			$drugIngredient{$drugname}{$ingredient}++;
		}
	} elsif ($drugname) {
		$drugcases{$drugname}{$indication}++;
		$totalcases{$drugname}++;
		$ndas{$drugname}{$nda}++ if $nda;
	} else {
		next;
	}
	#$done++;
	#last if $done >= 1e6;
}
close CL;

foreach my $drugname (sort {$totalcases{$b}<=>$totalcases{$a}} keys %drugcases) {
	my $useingredient = '';
	if (defined $ingcases{$drugname}) {
		# assume drugname and ingredient are the same
		#print "interpreting drugname $drugname as ingredient\n";
		$useingredient = $drugname;
	} elsif (defined $drugIngredient{$drugname}) {
		my @inglist = sort keys %{$drugIngredient{$drugname}};
		my %possible;
		foreach my $ingredient (@inglist) {
			if (defined $ingcases{$ingredient}) {
				$possible{$ingredient} = 1;
			}
		}
		my @possible = sort keys %possible;
		if (scalar @possible > 1) {
			print join("\t", "more than one possible match", $drugname, $totalcases{$drugname}, join("\$", @possible)), "\n";
		} else {
			print join("\t", "matched", $drugname, $totalcases{$drugname}, @possible), "\n";
		}
		$useingredient = $possible[0]; ### if there were multiple possible ingredients, this picks the first one... alphabetically, which makes little sense ###
	} else {
		print join("\t", "no ingredient", $drugname, $totalcases{$drugname}), "\n";
		$useingredient = $drugname; # keep the drug name as 'ingredient', in the hope that downstream mapping will recover it
	}
	# move content from drugcases to ingcases
	foreach my $indication (keys %{$drugcases{$drugname}}) {
		$ingcases{$useingredient}{$indication} += $drugcases{$drugname}{$indication};
	}
}

open OUTF, $outfile;
print OUTF join("\t", qw/CASES NDAS INGREDIENT INDICATION/), "\n";
foreach my $ingredient (sort keys %ingcases) {
	my $ndas = join(",", keys %{$ndas{$ingredient}});
	foreach my $indication (sort keys %{$ingcases{$ingredient}}) {
		print OUTF join("\t", $ingcases{$ingredient}{$indication}, $ndas, $ingredient || "??", $indication), "\n";
	}
}
close OUTF;

###
sub cleanText {
	my($txt) = @_;
	$txt =~ s/\s\s+/ /g;
	while ($txt =~ /^\"(.+)\"$/ || /^\'(.+)\'$/ || /^\s+(.+)$/ || /^(.+)\s+$/ || /^#+(.+)$/) {
		print "$txt -> $1\n";
		$txt = $1;
	}
	return lc $txt;
}
