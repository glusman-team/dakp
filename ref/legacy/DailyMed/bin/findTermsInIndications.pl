#!/bin/env perl
use strict;
require "../lib/libText.pl";

my $termMatchCutoff = 0.75;

my $mondoList = "../MONDO/results/mondo-terms.txt";
my $hpList = "../HPO/results/hp-terms.txt";
my $indications_file = "extracted/indications.txt";
my $word_count_file = "results/indication-words.txt";
my $stopwords_file = "../stopwords.txt";
my %idx;
my($mmeans, $malsomeans) = readTermList($mondoList);
my($hmeans, $halsomeans) = readTermList($hpList);
my $wordFreq = readWordFreq($word_count_file, $stopwords_file);
my($wordsInTerms, $maxScoreForTerm) = indexWordsInTerms($mmeans, $malsomeans, $hmeans, $halsomeans);



my($bin) = @ARGV;
my $cutoff = 0.85;
my %ind;
my $nlines;
my $cumulative;
my $matchedLines;
my $matchedCases;
if ($indications_file =~ /\.gz$/) {
	open IND, "gunzip -c $indications_file |";
} else {
	open IND, $indications_file;
}
LINE: while (<IND>) {
	$nlines++;
	#next unless ($nlines % 12) == $bin;
	chomp;
	my($xml, $originaltext) = split /\t/;
	my $text = lc $originaltext;
	$text =~ s/\<[^\>]*\>/ /g;
	#my $text = cleanText($originaltext);

	# split into words, shortlist terms to evaluate in depth
	my $todo = shortlistTerms($text);
	
	my %found;
	foreach my $term (keys %$todo) {
		my($match, $mode) = matchTerm($text, $term);
		
		if (!$match && $text =~ /(^|\s)cardiac(\s|$)/) {
			$text =~ s/cardiac/heart/g;
			($match, $mode) = matchTerm($text, $term);
		}
		if (!$match && $text =~ /(^|\s)renal(\s|$)/) {
			$text =~ s/renal/kidney/g;
			($match, $mode) = matchTerm($text, $term);
		}
		if (!$match && $text =~ /(^|\s)hepatic(\s|$)/) {
			$text =~ s/hepatic/liver/g;
			($match, $mode) = matchTerm($text, $term);
		}
		if (!$match && $text =~ /\'s(\s|$)/) {
			$text =~ s/'s//g;
			($match, $mode) = matchTerm($text, $term);
		}
		
		if ($match) {
			$found{$term} = $match;
		}
	}
	
	my %seen;
	foreach my $term (sort {$found{$a}<=>$found{$b}} keys %found) {
		my $meaning = $mmeans->{$term} || $malsomeans->{$term} || $hmeans->{$term} || $halsomeans->{$term};
		next unless $meaning;
		next if $seen{$meaning};
		print join("\t", $xml, $meaning, $found{$term}, $term), "\n";
		$seen{$meaning}++;
	}
	
}
close IND;

####
sub shortlistTerms {
	my($text) = @_;
	my %score;
	my %words;
	#print "$text\n\n";
	$words{$_}++ foreach split /\W+/, $text;
	my @words = keys %words;
	my %n;
	my %ratio;
	foreach my $word (@words) {
		foreach my $term (keys %{$wordsInTerms->{$word}}) {
			$score{$term} += $wordsInTerms->{$word}{$term};
		}
	}
	foreach my $term (keys %score) {
		my $nwords = scalar(split /\W+/, $term);
		my $ratio = $score{$term}/$nwords;
		next unless $ratio >= $termMatchCutoff;
		#print join("\t", $term, $terms{$term}, scalar keys %words, $maxScoreForTerm->{$term}, $ratio{$term}), "\n";
		$ratio{$term} = $ratio;
		#delete $terms{$term} if $ratio < 0.75;
	}
	return \%ratio;
}

#$means{$term} = join(",", sort keys %pids);
sub indexWordsInTerms {
	my %wit;
	my %max;
	foreach my $set (@_) {
		while (my($term, $ids) = each %$set) {
			next if defined $max{$term};
			my %words;
			$words{$_}++ foreach split /\W+/, $term;
			foreach (keys %words) {
				next unless $_;
				$max{$term} += ($wordFreq->{$_} || 1);
				$wit{$_}{$term} = ($wordFreq->{$_} || 1);
			}
		}
	}
	return \%wit, \%max;
}

sub simpleMatch {
	my($text, $term) = @_;
	
	my($pre, $post);
	my $len = length($text);
	my $where = index($text, $term);
	if ($where > -1) {
		$pre = substr($text, $where-1, 1) if $where;
		if ($where && $pre =~ /\W/) {
			my $after = $where+length($term);
			if ($after<$len) {
				$post = substr($text, $after, 1);
				if ($post =~ /\W/) {
					return $where+1;
				}
			}
		}
	}
	
	$text =~ s/oe/e/g;
	$text =~ s/ae/e/g;
	$text =~ s/ou/o/g;
	
	$where = index($text, $term);
	if ($where > -1) {
		$pre = substr($text, $where-1, 1) if $where;
		if ($where && $pre =~ /\W/) {
			my $after = $where+length($term);
			if ($after<$len) {
				$post = substr($text, $after, 1);
				if ($post =~ /\W/) {
					return $where+1;
				}
			}
		}
	}
	#return $where+1 if $where > -1;
}


sub matchTerm {
	my($text, $term) = @_;
	
	my $attempt = simpleMatch($text, $term);
	return $attempt if $attempt;
	
	my @words = split /[\s,]+/, $term;
	if (scalar @words > 1) {
		# try adding a comma between words
		for (my $i=0;$i<$#words;$i++) {
			my $withcomma = join(" ", map {$_==$i ? "$words[$_]," : $words[$_]} (0..$#words));
			$attempt = simpleMatch($text, $withcomma);
			return $attempt if $attempt;
		}
		
		# try moving words around
		for (my $i=0;$i<$#words;$i++) {
			my $rearranged1 = join(" ", @words[$i+1..$#words]);
			my $rearranged2 = join(" ", @words[0..$i]);
			$attempt = simpleMatch($text, join(" ", $rearranged1, $rearranged2));
			return $attempt if $attempt;
			$attempt = simpleMatch($text, join(", ", $rearranged1, $rearranged2));
			return $attempt if $attempt;
		}
	}
	
}

sub readWordFreq {
	my($file, $stopfile) = @_;
	my %stop;
	my %freq;
	
	open F, $stopfile;
	while (<F>) {
		next if /^#/;
		chomp;
		$stop{$_}++;
	}
	close F;
	
	open F, $file;
	while (<F>) {
		chomp;
		my($freq, $word) = split /\t/;
		if ($stop{$word}) {
			#print "skipping $word\n";
			$freq{$word} = 0.001;
		} else {
			$freq{$word} = 1-$freq;
		}
	}
	close F;
	return \%freq;
}




sub cleanText {
	my($txt) = @_;
	$txt =~ s/[\(\)\/]/ /g;
	$txt =~ s/[^a-z0-9 \'\-]+//gi;
	$txt =~ s/  +/ /g;
	$txt =~ s/^ //;
	$txt =~ s/ $//;
	return lc $txt;
}

sub readTermList {
	my($file) = @_;
	my(%means, %alsomeans);
	open MF, $file;
	$_ = <MF>; #headers
	while (<MF>) {
		chomp;
		my($name, $exact, $related, $narrow, $broad, $precise, $imprecise, $term) = split /\t/;
		$term = lc $term;
		#$term = cleanText($term);
		my $edited = $term;
		my $edited1 = $term;
		if ($term =~ /\'s( |$)/) {
			$edited =~ s/\'s//g;
			$edited1 =~ s/\'s/s/g;
			#print join("\t", $term, $edited), "\n";
		}
		my(%pids, %iids);
		if ($name ne 'NA') { $pids{$_}++ foreach split /,/, $name }
		if ($exact ne 'NA') { $pids{$_}++ foreach split /,/, $exact }
		if ($imprecise==1 && !$precise && $related && $related ne 'NA') {
			$pids{$related}++;
		} elsif ($related ne 'NA') {
			$iids{$_}++ foreach split /,/, $related;
		}
		$means{$term} = join(",", sort keys %pids);
		$means{$edited1} = $means{$edited} = $means{$term};
		if ($narrow ne 'NA') { $iids{$_}++ foreach split /,/, $narrow }
		if ($broad ne 'NA') { $iids{$_}++ foreach split /,/, $broad }
		$alsomeans{$term} = join(",", sort keys %iids);
		$alsomeans{$edited1} = $alsomeans{$edited} = $alsomeans{$term};
	}
	return \%means, \%alsomeans;
}
