#!/bin/env perl
use strict;
require "../lib/libSystem.pl";
require "../lib/libText.pl";
$|=1;

my($indir, $casesOutfile, $indicationsOutfile, $drugsOutfile) = @ARGV;
$indir ||= "data";
$casesOutfile ||= "results/cases";
$indicationsOutfile ||= "results/indications";
$drugsOutfile ||= "results/drugs";
my $ignoreSpecial = 0;

my %ignore = (
	#'product used for unknown indication', 1,
	#'product used for unknown indcation', 1,
	#'drug used for unknown indication', 1,
	#'drug use for unknown indication', 1,
	#'drug use fo runknown indication', 1,
	'drug use unknown indication', 1,
	#'off label use', 1,
	'prophylaxis', 1,
	'ill-defined disorder', 1,
	'premedication', 1,
	#'product use in unapproved indication', 1,
	'intentional product misuse', 1,
	'exposure during pregnancy', 1,
	'foetal exposure during pregnancy', 1,
	'product origin unknown', 1,
	#'drug known for unknown indication', 1,
	#'unknown route of product administration', 1,
	#'unknown schedule of product administration', 1,
	'accidental exposure to product', 1,
	'product use issue', 1,
);

my $delete = readDELETE($indir); # cases to be ignored
my @quarters = findQuarters($indir); # available data quarters

my %prevSeenCase;
my %drugName_indi;
my %drugNDA_indi;
my %nda;
my %ndadrugs;
my %indiCases;
my %drugCases;

open OUTF, "| gzip -c >$casesOutfile.txt.gz";
print OUTF join("\t", qw/quarter primaryid caseid source occp_cod reporter_country drugname ingredient nda indication effects/), "\n";
foreach my $q (@quarters) {
	# read demographics
	my($demo) = readDEMO("$indir/DEMO$q.txt.gz");
	
	# read drug info
	my($case_drugInfo, $seenInQ) = readDRUG("$indir/DRUG$q.txt.gz");
	
	# read reporting sources
	my($source) = readRPSR("$indir/RPSR$q.txt.gz");
	
	# read side effects
	my($reaction) = readREAC("$indir/REAC$q.txt.gz");
	
	# finally, process indications, and print combined content to outfile
	readINDI("$indir/INDI$q.txt.gz", $case_drugInfo, $seenInQ, $q, $demo, $source, $reaction);

	# track already seen cases
	while (my($id) = each %$seenInQ) {
		$prevSeenCase{$id} = 1;
	}
}
close OUTF;

my %indications;
while (my($ind, $c) = each %indiCases) {
	$indications{$ind} = scalar keys %$c;
}

open OUTF, ">$indicationsOutfile.txt";
foreach my $ind (sort {$indications{$b}<=>$indications{$a}} keys %indications) {
	print OUTF join("\t", $indications{$ind}, $ind), "\n";
}
close OUTF;

my %drugs;
while (my($ind, $c) = each %drugCases) {
	$drugs{$ind} = scalar keys %$c;
}

open OUTF, ">$drugsOutfile.txt";
foreach my $drug (sort {$drugs{$b}<=>$drugs{$a}} keys %drugs) {
	print OUTF join("\t", $drugs{$drug}, $drug), "\n";
}
close OUTF;


###
sub readDELETE {
	my($dir) = @_;
	my %info;
	foreach my $file (slicedirlist($dir, "^DELETE")) {
		open F, "gunzip -c $dir/$file |";
		while (<F>) {
			chomp;
			s/\r//g;
			$info{$_}++;
		}
		close F;
	}
	return \%info;
}

sub findQuarters {
	my($dir) = @_;
	my %info;
	foreach my $file (fulldirlist($indir)) {
		my($q) = $file =~ /(\d\dQ\d)/i;
		$info{$q}++ if $q;
	}
	return reverse sort keys %info;
}

sub readDEMO {
	my($file) = @_;
	my %info;
	my @fields = qw/rept_dt occp_cod reporter_country/;
	open F, "gunzip -c $file |";
	$_ = lc <F>;
	chomp;
	s/\r//g;
	my @headers = split /\$/;
	my %col;
	$col{$headers[$_]} = $_ foreach (0..$#headers);
	while (<F>) {
		chomp;
		s/\r//g;
		my @v = split /\$/;
		my $primary = $v[$col{'primaryid'} // $col{'isr'}];
		next if $delete->{$primary};
		foreach my $field (@fields) {
			$info{$primary}{$field} = $v[$col{$field}];
		}
	}
	close F;
	return \%info;
}

sub readDRUG {
	my($file) = @_;
	my %drugInfo;
	my %seenInQ;
	open F, "gunzip -c $file |";
	$_ = lc <F>;
	chomp;
	s/\r//g;
	my @headers = split /\$/;
	my %col;
	$col{$headers[$_]} = $_ foreach (0..$#headers);
	while (<F>) {
		chomp;
		s/\r//g;
		my @v = split /\$/;
		my $primary = $v[$col{'primaryid'} // $col{'isr'}];
		next if $delete->{$primary};
		next if $prevSeenCase{$primary};
		my $caseid = $v[$col{'caseid'}];
		if ($caseid) {
			next if $prevSeenCase{$caseid};
			$seenInQ{$caseid} = 1;
		} else {
			$seenInQ{$primary} = 1;
		}
		my $drugname = polishText($v[$col{'drugname'}]);
		$drugname =~ s/\.$//;
		my $drugseq = $v[$col{'drug_seq'}] if defined $col{'drug_seq'};
		# role_cod field: PS primary suspect, SS secondary suspect, C concomitant, I interacting
		my $ingredient = polishText($v[$col{'prod_ai'}]) if defined $col{'prod_ai'};
		my $nda = $v[$col{'nda_num'}];
		my $case = "$primary\$$drugseq";
		$drugInfo{$case} = join("\t", $drugname, $ingredient, $nda);
	}
	close F;
	return \%drugInfo, \%seenInQ;
}

sub readRPSR {
	my($file) = @_;
	my %info;
	open F, "gunzip -c $file |";
	$_ = lc <F>;
	chomp;
	s/\r//g;
	my @headers = split /\$/;
	my %col;
	$col{$headers[$_]} = $_ foreach (0..$#headers);
	while (<F>) {
		chomp;
		s/\r//g;
		my @v = split /\$/;
		my $primary = $v[$col{'primaryid'} // $col{'isr'}];
		next if $delete->{$primary};
		my $caseid = $v[$col{'caseid'}];
		my $rpsr = $v[$col{'rpsr_cod'}];
		$rpsr =~ s/\s$//;
		$info{$primary} = $rpsr;
	}
	close F;
	return \%info;
}

sub readREAC {
	my($file) = @_;
	my %info;
	open F, "gunzip -c $file |";
	$_ = lc <F>;
	chomp;
	s/\r//g;
	my @headers = split /\$/;
	my %col;
	$col{$headers[$_]} = $_ foreach (0..$#headers);
	while (<F>) {
		chomp;
		s/\r//g;
		my @v = split /\$/;
		my $primary = $v[$col{'primaryid'} // $col{'isr'}];
		next if $delete->{$primary};
		$info{$primary}{$v[$col{'pt'}]}++;
	}
	close F;
	return \%info;
}

sub readINDI {
	my($file, $case_drug, $seenInQ, $q, $demo, $source, $reaction) = @_;
	my %seenRow;
	open F, "gunzip -c $file |";
	$_ = lc <F>;
	chomp;
	s/\r//g;
	my @headers = split /\$/;
	my %col;
	$col{$headers[$_]} = $_ foreach (0..$#headers);
	while (<F>) {
		chomp;
		s/\r//g;
		my @v = split /\$/;
		my $primary = $v[$col{'primaryid'} // $col{'isr'}];
		next if $delete->{$primary};
		my $caseid = $v[$col{'caseid'}];
		if ($caseid) {
			next unless $seenInQ->{$caseid};
		} else {
			next unless $seenInQ->{$primary};
		}
		my $drugseq = $v[$col{'indi_drug_seq'} // $col{'drug_seq'}];
		my $indication = $v[$col{'indi_pt'}];

		my $clean = polishText($indication);
		if ($ignoreSpecial) {
			next if $ignore{$indication};
			if ($ignore{$clean} || $clean =~ /unknown indication|product administration|unknown indcation/) {
				$ignore{$indication} = 1;
				next;
			}
		}
		
		my $case = "$primary\$$drugseq";
		my $drug = $case_drug->{$case};
		next unless $drug;
		my $effect;
		$effect = join("\$", sort keys %{$reaction->{$primary}}) if defined $reaction->{$primary};
		my $row = join("\t", $q, $primary, $caseid, $source->{$primary}, $demo->{$primary}{'occp_cod'}, $demo->{$primary}{'reporter_country'}, $drug, $clean, $effect);

		# some cases have multiple reports of what appears to be the same thing - deduplicate!
		next if $seenRow{$row};
		$seenRow{$row} = 1;

		print OUTF $row, "\n";
		$indiCases{$clean}{$caseid}++;
		$drugCases{$drug}{$caseid}++;
	}
	close F;
}
