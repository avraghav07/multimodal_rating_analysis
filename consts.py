# For converting rating to numerical
rating_order = [
        'AAA', 'AA+', 'AA', 'A+', 'A',
        'BBB+', 'BBB', 'BB+', 'BB',
        'B+', 'B'
    ]
rating_map = {rating: idx for idx, rating in enumerate(rating_order)}

# For keyword features
positive_words = ['growth', 'increase', 'improve', 'strong', 'positive', 'gain', 'profit', 
                  'expand', 'efficient', 'success', 'opportunity']
negative_words = ['decline', 'loss', 'challenge', 'risk', 'weak', 'decrease', 'problem',
                  'concern', 'difficult', 'struggle', 'volatility']