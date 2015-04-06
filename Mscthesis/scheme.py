
"""
Module which scheme information about the dataset.

"""

identifiers = ['Nom, nif']
location_data = ['cp', 'localidad', 'X', 'Y', 'ES-X', 'ES-Y']
coordinates_data = ['ES-X', 'ES-Y']
pol_loc_data = ['cp', 'location_data']
fecha_cierre = ['cierre']

#### Temporal data ####
total_activo = ['06Act', '07Act', '08Act', '09Act', '10Act', '11Act', '12Act']
activo_circu = ['06ActC', '07ActC', '08ActC', '09ActC', '10ActC', '11ActC',
                '12ActC']
pasivo_fijo = ['06Pasfijo', '07Pasfijo', '08Pasfijo', '09Pasfijo', '10Pasfijo',
               '11Pasfijo', '12Pasfijo']
pasivo_liqui = ['06Pasliq', '07Pasliq', '08Pasliq', '09Pasliq', '10Pasliq',
                '11Pasliq', '12Pasliq']
num_empleados = ['06Trab', '07Trab', '08Trab', '09Trab', '10Trab', '11Trab',
                 '12Trab']
valor_agg = ['06Va', '07Va', '08Va', '09Va', '10Va', '11Va', '12Va']
importe_ventas = ['06Vtas', '07Vtas', '08Vtas', '09Vtas', '10Vtas', '11Vtas',
                  '12Vtas']

finantial_vars = [total_activo, activo_circu, pasivo_fijo, pasivo_liqui,
                  num_empleados, valor_agg, importe_ventas]
